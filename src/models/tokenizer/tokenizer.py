"""
Credits to https://github.com/CompVis/taming-transformers
"""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, Optional

import einops
from loguru import logger
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataset import Batch
from .lpips import LPIPS
from utils import (
    LossWithIntermediateLosses, VectorQuantizer, ObsModality, FreqDist
)
from utils.math import base_n_to_base_10, base_10_to_base_n


@dataclass
class TokenizerEncoderOutput:
    z: Tensor
    z_quantized: Tensor
    tokens: Tensor


class TokenizerBase(ABC, nn.Module):
    @property
    @abstractmethod
    def modality(self) -> ObsModality:
        pass

    @property
    @abstractmethod
    def is_trainable(self) -> bool:
        pass

    @property
    @abstractmethod
    def tokens_per_obs(self) -> int:
        pass

    @abstractmethod
    def forward(self, x: Tensor, should_preprocess: bool = False, should_postprocess: bool = False,
                return_tokens: bool = False) -> Tuple[Tensor, ...]:
        pass

    @abstractmethod
    def compute_loss(self, batch: Batch, **kwargs: Any) -> tuple[LossWithIntermediateLosses, dict]:
        pass

    @abstractmethod
    def encode(self, x: Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        pass

    @abstractmethod
    def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
        pass

    @abstractmethod
    def to_codes(self, tokens, **kwargs):
        pass

    @torch.no_grad()
    def encode_decode(self, x: Tensor, should_preprocess: bool = False,
                      should_postprocess: bool = False) -> Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)


def _combine_encoder_outputs(outputs: list[TokenizerEncoderOutput], input_shape=None) -> TokenizerEncoderOutput:
    assert len(outputs) > 0
    results = TokenizerEncoderOutput(
        z=torch.cat([r_i.z for r_i in outputs], dim=0),
        z_quantized=torch.cat([r_i.z_quantized for r_i in outputs], dim=0),
        tokens=torch.cat([r_i.tokens for r_i in outputs], dim=0),
    ) if len(outputs) > 1 else outputs[0]
    if input_shape is not None:
        results.z = results.z.reshape(*input_shape[:-3], *results.z.shape[1:])
        results.z_quantized = results.z_quantized.reshape(*input_shape[:-3], *results.z_quantized.shape[1:])
        results.tokens = results.tokens.reshape(*input_shape[:-3], *results.tokens.shape[1:])
    return results


class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al., ICLR 2024).

    Each latent dimension is independently quantized to L_i evenly-spaced integer levels.
    No learned codebook, no EMA, no commitment loss — codebook utilization is near-100%
    by construction.

    Example: levels=[8, 8, 8] → vocab_size = 8³ = 512, num_dims = 3.
    """

    def __init__(self, levels: list[int]) -> None:
        super().__init__()
        levels_t = torch.tensor(levels, dtype=torch.long)
        basis = torch.cumprod(
            torch.cat([torch.ones(1, dtype=torch.long), levels_t[:-1]]), dim=0
        )
        self.register_buffer('_levels', levels_t)
        self.register_buffer('_basis', basis)
        self.num_dims = len(levels)
        self.vocab_size = int(levels_t.prod().item())

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        eps = 1e-3
        half_l = (self._levels.float() - 1) * (1 - eps) / 2
        offset = torch.where(
            self._levels % 2 == 0,
            torch.full_like(self._levels, 0.5, dtype=torch.float),
            torch.zeros_like(self._levels, dtype=torch.float),
        )
        shift = torch.tan(offset / half_l)
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize with straight-through estimator. z: (..., num_dims)"""
        z_b = self._bound(z)
        quantized = z_b + (z_b.round() - z_b).detach()

        # renormalize to [-1, 1]:
        half_width = self._levels // 2
        return quantized / half_width

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Normalized codes (..., num_dims) in [-1, 1] → integer token indices (...)."""
        half_l = self._levels // 2
        int_codes = (codes * half_l.float() + half_l.float()).round().long()
        return (int_codes * self._basis).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Integer token indices (...) → normalized codes (..., num_dims) in [-1, 1]."""
        half_l = self._levels // 2
        int_codes = (indices.unsqueeze(-1) // self._basis) % self._levels
        return (int_codes.float() - half_l.float()) / half_l.float()

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (..., num_dims) → (quantized_codes (..., num_dims), indices (...))"""
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        return codes, indices


class ImageTokenizer(TokenizerBase):
    def __init__(
            self, vocab_size: int, embed_dim: int, vgg_lpips_ckpt_path: str, encoder: nn.Module,
            decoder: nn.Module, with_lpips: bool = True, device=None,
            fsq_levels: Optional[list[int]] = None,
            # VQ-only params (ignored when fsq_levels is set):
            ema_decay: float = 0.99, commitment_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.lpips = LPIPS(vgg_lpips_ckpt_path).eval().to(device) if with_lpips else None

        if fsq_levels is not None:
            d = len(fsq_levels)
            self.fsq = FSQ(list(fsq_levels)).to(device)
            self.pre_quant_conv = nn.Conv2d(embed_dim, d, 1, device=device)
            self.post_quant_conv = nn.Conv2d(d, embed_dim, 1, device=device)
            self._vocab_size = self.fsq.vocab_size
            self.embedding = None
        else:
            self.fsq = None
            self.pre_quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()
            self.embedding = nn.Embedding(vocab_size, embed_dim, device=device)
            self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
            self._vocab_size = vocab_size
            self._ema_decay = ema_decay
            self._commitment_beta = commitment_beta
            if ema_decay > 0:
                self.register_buffer('_ema_cluster_size', torch.zeros(vocab_size, device=device))
                self.register_buffer('_ema_embed_avg', self.embedding.weight.data.clone())

        self._effective_bsz = None
        self._past_err_msgs = []

    def __repr__(self) -> str:
        return "tokenizer"

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def modality(self) -> ObsModality:
        return ObsModality.image

    @property
    def is_trainable(self) -> bool:
        return True

    @property
    def tokens_per_obs(self) -> int:
        res = np.array(self.encoder.config.input_resolution)
        return int(np.prod(res / 2 ** self.encoder.config.num_downsample_steps))

    def forward(self, x: Tensor, should_preprocess: bool = False, should_postprocess: bool = False,
                return_tokens: bool = False) -> Tuple[Tensor, ...]:
        outputs = self.encode(x, should_preprocess)  # z and z_quantized are both embed_dim
        if self.fsq is not None:
            reconstructions = self.decode(outputs.z_quantized, should_postprocess)
        else:
            # VQ straight-through estimator
            decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
            reconstructions = self.decode(decoder_input, should_postprocess)
        if return_tokens:
            return outputs.z, outputs.z_quantized, reconstructions, outputs.tokens
        return outputs.z, outputs.z_quantized, reconstructions

    def _auto_adjust_bsz_call(self, x: Tensor, fn, combine_results_fn, **kwargs):
        """Split oversized batches into sequential mini-batches to avoid OOM."""
        input_shape = x.shape
        x = x.reshape(-1, *input_shape[-3:])
        input_bsz = x.shape[0]
        bsz = input_bsz if self._effective_bsz is None else self._effective_bsz
        while bsz > 0:
            try:
                num_mini_batches = math.ceil(input_bsz / bsz)
                results = [fn(x[i * bsz:(i+1) * bsz], **kwargs) for i in range(num_mini_batches)]
                results = combine_results_fn(results, input_shape)
                if bsz < input_bsz and self._effective_bsz is None:
                    self._effective_bsz = bsz
                return results
            except torch.OutOfMemoryError:
                err_msg = f"Out of Memory with batch size = {bsz}, trying {bsz // 2}..."
                if err_msg not in self._past_err_msgs:
                    self._past_err_msgs.append(err_msg)
                    logger.warning(err_msg)
                bsz = bsz // 2
        raise RuntimeError('No batch size fits the available memory!')

    def compute_loss(self, batch: Batch, **kwargs: Any) -> tuple[LossWithIntermediateLosses, dict]:
        obs = batch['observations'][ObsModality.image]
        assert obs.shape[1] == 1
        observations = self.preprocess_input(rearrange(obs, 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions, tokens = self(
            observations, should_preprocess=False, should_postprocess=False, return_tokens=True
        )

        if self.fsq is not None:
            loss_dict = {}
        else:
            z_flat = rearrange(z, 'b e h w -> (b h w) e')
            z_q_flat = rearrange(z_quantized, 'b e h w -> (b h w) e')
            tokens_flat = tokens.reshape(-1)

            if self._ema_decay > 0 and self.training:
                with torch.no_grad():
                    one_hot = F.one_hot(tokens_flat, self._vocab_size).float()
                    new_cluster_size = one_hot.sum(0)
                    new_embed_sum = one_hot.t() @ z_flat

                    self._ema_cluster_size.mul_(self._ema_decay).add_(new_cluster_size * (1 - self._ema_decay))
                    self._ema_embed_avg.mul_(self._ema_decay).add_(new_embed_sum * (1 - self._ema_decay))

                    n = self._ema_cluster_size.sum()
                    smoothed = (self._ema_cluster_size + 1e-5) / (n + self._vocab_size * 1e-5) * n
                    self.embedding.weight.data.copy_(self._ema_embed_avg / smoothed.unsqueeze(1))

                    dead = self._ema_cluster_size < 1.0
                    num_dead = int(dead.sum().item())
                    if num_dead > 0:
                        rand_idx = torch.randint(0, z_flat.shape[0], (num_dead,), device=z_flat.device)
                        self.embedding.weight.data[dead] = z_flat[rand_idx].detach()

                commitment_loss = self._commitment_beta * F.mse_loss(z_flat, z_q_flat.detach())
            else:
                commitment_loss = (
                    F.mse_loss(z_q_flat, z_flat.detach()) +
                    self._commitment_beta * F.mse_loss(z_flat, z_q_flat.detach())
                )
            loss_dict = dict(commitment_loss=commitment_loss)

        reconstruction_loss = F.mse_loss(observations, reconstructions)
        if self.lpips is not None:
            self.lpips.eval()
            perceptual_loss = torch.mean(self.lpips(observations, reconstructions).flatten())
            loss_dict['perceptual_loss'] = perceptual_loss

        loss_dict['reconstruction_loss'] = reconstruction_loss

        with torch.no_grad():
            info = {
                'token_counts': FreqDist(tokens.detach().flatten().cpu().numpy()),
            }

        return LossWithIntermediateLosses(**loss_dict), info

    def encode(self, x: Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        return self._auto_adjust_bsz_call(x, self._encode, _combine_encoder_outputs,
                                          should_preprocess=should_preprocess)

    def _encode(self, x: Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)  # (B, embed_dim, h, w) — stored as-is for uniform interface

        if self.fsq is not None:
            b, _, h, w = z.shape
            z_d = self.pre_quant_conv(z)  # (B, d, h, w)
            codes, indices = self.fsq(rearrange(z_d, 'b d h w -> b h w d'))
            z_q = self.post_quant_conv(rearrange(codes, 'b h w d -> b d h w').contiguous())  # (B, embed_dim, h, w)
            tokens = indices.reshape(b, -1)
        else:
            b, e, h, w = z.shape
            z_flat = rearrange(z, 'b e h w -> (b h w) e')
            dist = (torch.sum(z_flat ** 2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight ** 2, dim=1)
                    - 2 * z_flat @ self.embedding.weight.t())
            tokens = dist.argmin(dim=-1)
            z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, h=h, w=w).contiguous()
            tokens = tokens.reshape(b, -1)

        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)
        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
        shape = z_q.shape  # (..., embed_dim, h, w)
        z_q = z_q.reshape(-1, *shape[-3:])
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec

    @torch.no_grad()
    def to_codes(self, tokens: Tensor, **kwargs) -> Tensor:
        hw = tokens.shape[-1]
        h = w = int(np.sqrt(hw))
        if self.fsq is not None:
            codes = self.fsq.indices_to_codes(tokens)           # (..., hw, d)
            z_q = rearrange(codes, '... (h w) d -> ... d h w', h=h, w=w).contiguous()
            shape = z_q.shape
            z_q = self.post_quant_conv(z_q.reshape(-1, *shape[-3:]))  # d → embed_dim
            return z_q.reshape(*shape[:-3], *z_q.shape[-3:])
        else:
            emb = self.embedding(tokens)                        # (..., hw, embed_dim)
            return rearrange(emb, '... (h w) e -> ... e h w', h=h, w=w).contiguous()

    @torch.no_grad()
    def encode_decode(self, x: Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: Tensor) -> Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: Tensor) -> Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)


class HardCodedVectorTokenizer(TokenizerBase):

    def __init__(self, input_dim: int, vector_quantizer: VectorQuantizer = None, device=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.vector_quantizer = VectorQuantizer(normalize=False) if vector_quantizer is None else vector_quantizer
        # self.vector_quantizer = ByteToken2FP16Mapper()
        if device is not None:
            self.vector_quantizer.to(device)

    @property
    def modality(self) -> ObsModality:
        return ObsModality.vector

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def tokens_per_obs(self) -> int:
        return self.input_dim

    @property
    def vocab_size(self):
        return self.vector_quantizer.vocab_size

    def forward(self, x: Tensor, should_preprocess: bool = False, should_postprocess: bool = False,
                return_tokens: bool = False) -> Tuple[Tensor, ...]:
        outputs = self.encode(x, should_preprocess)
        reconstructions = outputs.z_quantized
        if return_tokens:
            return outputs.z, outputs.z_quantized, reconstructions, outputs.tokens
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> tuple[LossWithIntermediateLosses, dict]:
        return LossWithIntermediateLosses(), {}

    def encode(self, x: Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        z = x
        tokens = self.vector_quantizer.vector_to_tokens_pt(x)
        z_q = self.vector_quantizer.tokens_to_vector_pt(tokens)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
        return z_q

    def to_codes(self, tokens, **kwargs):
        return self.vector_quantizer.tokens_to_vector_pt(tokens)


class FSQVectorTokenizer(TokenizerBase):
    """MLP encoder → split → FSQ per chunk → MLP decoder for continuous vector obs (e.g. DMC).

    The encoder projects to (num_tokens * num_fsq_dims), which is split into num_tokens chunks
    and each chunk is independently quantized by FSQ, yielding num_tokens tokens per observation.
    """

    def __init__(self, input_dim: int, levels: list[int], num_tokens: int = 8,
                 hidden_dim: int = 256, num_layers: int = 2, device=None) -> None:
        super().__init__()
        self._num_tokens = num_tokens
        d = len(levels)

        def mlp(in_dim, out_dim):
            layers = []
            cur = in_dim
            for _ in range(num_layers):
                layers += [
                    nn.Linear(cur, hidden_dim, device=device),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim, device=device)
                ]
                cur = hidden_dim
            layers.append(nn.Linear(cur, out_dim, device=device))
            return nn.Sequential(*layers)

        self.encoder = mlp(input_dim, num_tokens * d)
        self.decoder = mlp(num_tokens * d, input_dim)
        self.fsq = FSQ(levels)
        if device is not None:
            self.fsq = self.fsq.to(device)

    def __repr__(self):
        return 'FSQVectorTokenizer'

    @property
    def modality(self) -> ObsModality:
        return ObsModality.vector

    @property
    def is_trainable(self) -> bool:
        return True

    @property
    def tokens_per_obs(self) -> int:
        return self._num_tokens

    @property
    def vocab_size(self) -> int:
        return self.fsq.vocab_size

    def forward(self, x: Tensor, should_preprocess: bool = False, should_postprocess: bool = False,
                return_tokens: bool = False) -> Tuple[Tensor, ...]:
        outputs = self.encode(x, should_preprocess)
        reconstructions = self.decode(outputs.z_quantized, should_postprocess)
        if return_tokens:
            return outputs.z, outputs.z_quantized, reconstructions, outputs.tokens
        return outputs.z, outputs.z_quantized, reconstructions

    def preprocess_input(self, x: Tensor) -> Tensor:
        from utils.math import sym_log
        return sym_log(x.float())

    def postprocess_output(self, z: Tensor) -> Tensor:
        from utils.math import sym_exp
        return sym_exp(z.float())

    def compute_loss(self, obs: Tensor, **kwargs: Any) -> tuple[LossWithIntermediateLosses, dict]:
        # obs: (B, T, D) raw float32
        assert obs.shape[1] == 1
        z, z_quantized, reconstructions, tokens = self(
            obs, should_preprocess=True, should_postprocess=False, return_tokens=True
        )
        reconstruction_loss = F.mse_loss(self.preprocess_input(obs), reconstructions)
        with torch.no_grad():
            info = {'token_counts': FreqDist(tokens.detach().flatten().cpu().numpy())}
        return LossWithIntermediateLosses(reconstruction_loss=reconstruction_loss), info

    def encode(self, x: Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        shape = x.shape  # (..., input_dim)
        d = self.fsq.num_dims
        if should_preprocess:
            x = self.preprocess_input(x)
        z = self.encoder(x.reshape(-1, shape[-1]))                        # (B, num_tokens * d)
        z_split = z.reshape(-1, self._num_tokens, d)                      # (B, num_tokens, d)
        codes, indices = self.fsq(z_split)                                # (B, num_tokens, d), (B, num_tokens)
        z_q = codes.reshape(*shape[:-1], self._num_tokens, d)             # (..., num_tokens, d)
        z = z.reshape(*shape[:-1], self._num_tokens, d)
        tokens = indices.reshape(*shape[:-1], self._num_tokens)           # (..., num_tokens)
        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
        shape = z_q.shape  # (..., num_tokens, d)
        flat = z_q.reshape(-1, self._num_tokens * self.fsq.num_dims)     # (B, num_tokens * d)
        x_hat = self.decoder(flat).reshape(*shape[:-2], -1)               # (..., input_dim)
        if should_postprocess:
            x_hat = self.postprocess_output(x_hat)
        return x_hat

    def to_codes(self, tokens: Tensor, **kwargs) -> Tensor:
        # tokens: (..., num_tokens) → codes: (..., num_tokens, d)
        return self.fsq.indices_to_codes(tokens)


class DummyTokenizer(TokenizerBase):

    def __init__(self, nvec: np.ndarray, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nvec = nvec
        if nvec.ndim == 1:
            self._modality = ObsModality.token
        else:
            assert nvec.ndim == 2, f"{nvec.ndim}-dim is not supported"
            self._modality = ObsModality.token_2d

    @property
    def modality(self) -> ObsModality:
        return self._modality

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def tokens_per_obs(self) -> int:
        return self.nvec.shape[0]

    @property
    def vocab_size(self):
        return self.nvec[0]

    def forward(self, x: Tensor, should_preprocess: bool = False, should_postprocess: bool = False,
                return_tokens: bool = False) -> Tuple[Tensor, ...]:
        return x, x, x

    def compute_loss(self, batch: Batch, **kwargs: Any) -> tuple[LossWithIntermediateLosses, dict]:
        return LossWithIntermediateLosses(), {}

    def encode(self, x: Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        return TokenizerEncoderOutput(x, x, x)

    def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
        return z_q

    def to_codes(self, tokens, **kwargs):
        return tokens

    def encode_decode(self, x: Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tensor:
        return super().encode_decode(x, should_preprocess, should_postprocess)


class MultiModalTokenizer(nn.Module):
    def __init__(
            self,
            tokenizers: dict[ObsModality, TokenizerBase],
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tokenizers = nn.ModuleDict({k.name: v for k, v in tokenizers.items()})

    def __repr__(self):
        return 'tokenizers'

    @property
    def modalities(self) -> set[ObsModality]:
        return set([t.modality for t in self.tokenizers.values()])

    @property
    def is_trainable(self) -> bool:
        return any([t.is_trainable for t in self.tokenizers.values()])

    @property
    def tokens_per_obs(self) -> int:
        return sum([t.tokens_per_obs for t in self.tokenizers.values()])

    @property
    def tokens_per_obs_dict(self) -> dict[ObsModality, int]:
        return {ObsModality[k]: v.tokens_per_obs for k, v in self.tokenizers.items()}

    @property
    def vocab_size(self) -> dict[ObsModality, int]:
        return {ObsModality[k]: v.vocab_size for k, v in self.tokenizers.items()}

    def forward(
            self,
            x: dict[ObsModality, Tensor],
            should_preprocess: bool = False,
            should_postprocess: bool = False,
            return_tokens: bool = False
    ) -> dict[ObsModality, Tuple[Tensor]]:
        assert set(x.keys()) == set([ObsModality[k] for k in self.tokenizers.keys()]), \
            f"Obs keys ({x.keys()}) != tokenizers keys ({self.tokenizers.keys()})"
        return {
            ObsModality[k]: self.tokenizers[k].forward(
                x[ObsModality[k]], should_preprocess, should_postprocess, return_tokens)
            for k in self.tokenizers.keys()
        }

    def compute_loss(self, batch: Batch, **kwargs: Any) -> tuple[LossWithIntermediateLosses, dict]:
        losses = {ObsModality[k]: self.tokenizers[k].compute_loss(batch, **kwargs) for k in self.tokenizers.keys()}
        combined = LossWithIntermediateLosses.combine([l[0] for l in losses.values()])
        infos = {k.name: l[1] for k, l in losses.items()}

        return combined, infos

    def encode(
            self,
            x: dict[ObsModality, Tensor],
            should_preprocess: bool = False
    ) -> dict[ObsModality, TokenizerEncoderOutput]:
        assert set(x.keys()) == set([ObsModality[k] for k in self.tokenizers.keys()]), \
            f"Obs keys ({x.keys()}) != tokenizers keys ({self.tokenizers.keys()})"
        return {ObsModality[k]: self.tokenizers[k].encode(x[ObsModality[k]], should_preprocess)
                for k in self.tokenizers.keys()}

    def decode(
            self,
            z_q: dict[ObsModality, Tensor],
            should_postprocess: bool = False
    ) -> dict[ObsModality, Tensor]:
        return {ObsModality[k]: self.tokenizers[k].decode(z_q[ObsModality[k]], should_postprocess)
                for k in self.tokenizers.keys()}

    def to_codes(self, tokens: dict[ObsModality, Tensor], **kwargs) -> dict[ObsModality, Tensor]:
        return {ObsModality[k]: self.tokenizers[k].to_codes(tokens[ObsModality[k]], **kwargs)
                for k in self.tokenizers.keys()}

    def encode_decode(self, x: dict[ObsModality, Tensor], should_preprocess: bool = False,
                      should_postprocess: bool = False) -> dict[ObsModality, Tensor]:
        encoded = self.encode(x, should_preprocess=should_preprocess)
        return self.decode({k: v.z_quantized for k, v in encoded.items()}, should_postprocess=should_postprocess)
