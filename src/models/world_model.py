from abc import abstractmethod, ABC
from dataclasses import dataclass

import numpy as np
from math import ceil
from typing import Any, Optional, Union

from loguru import logger
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataset import Batch

from .pop_retnet import POPRetNetDecoderLayer, POPRetNetDecoder
from .tokenizer import HardCodedVectorTokenizer, MultiModalTokenizer
from utils import (
    LossWithIntermediateLosses, RecurrentState, UniformVQ, sym_exp,
    HLGaussCategoricalRegressionHead
)
from utils.types import MultiModalObs, ObsModality
from utils.distributions import sample_categorical, MultiCategoricalSampler
from models.embedding import make_embeddings, ActionEncoder


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


@dataclass
class RetNetConfig:
    max_blocks: int

    num_layers: int
    num_heads: int
    embed_dim: int

    dropout: float

    blocks_per_chunk: int

    mask_type: str

    decay_scale_min_num_blocks: int
    decay_scale_max_num_blocks: int

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks

    @property
    def tokens_per_chunk(self):
        return self.blocks_per_chunk * self.tokens_per_block


from torch.func import stack_module_state, functional_call
from torch import vmap


def get_vocab_head_dim(obs_vocab_size: Union[int, np.ndarray]):
    # determine the output size:
    if isinstance(obs_vocab_size, np.ndarray):
        assert obs_vocab_size.ndim == 1
        vocab_dims = np.sum(obs_vocab_size)
    else:
        assert isinstance(obs_vocab_size, int) or np.isscalar(obs_vocab_size)
        vocab_dims = obs_vocab_size
    return vocab_dims


def default_ce_loss(logits, labels, vocab_size=None, reduction='mean'):
    return F.cross_entropy(logits, labels, reduction=reduction)


def token_2d_ce_loss(logits, labels, vocab_size: np.ndarray, reduction='mean'):
    assert isinstance(vocab_size, np.ndarray) and vocab_size.ndim == 1
    sum_dims = vocab_size.sum()
    assert logits.shape[-1] == sum_dims, f'got {logits.shape}[-1] != {sum_dims}'

    separate_logits = torch.split(logits, vocab_size.tolist(), dim=-1)
    separate_labels = torch.split(labels.reshape(-1, vocab_size.size), 1, dim=-1)

    loss = torch.stack([
        F.cross_entropy(logits_i, labels_i.flatten(), reduction=reduction)
        for logits_i, labels_i in zip(separate_logits, separate_labels)
    ], dim=-1).sum(-1)

    return loss


modality_to_obs_loss = {
    ObsModality.image: default_ce_loss,
    ObsModality.vector: default_ce_loss,
    ObsModality.token: default_ce_loss,
    ObsModality.token_2d: token_2d_ce_loss,
}


class EnsembleObsHead(nn.Module):

    def __init__(self, ensemble_size: int, embed_dim: int, vocab_size: int, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_size = ensemble_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.ensemble = nn.ModuleList([
            self._build_model(device=device) for _ in range(ensemble_size)
        ])

        self.meta_model = None
        self.ens_params = None
        self.ens_buffers = None

        self.init_meta_model()

    def init_meta_model(self):
        self.meta_model = self._build_model().to('meta')
        self.ens_params, self.ens_buffers = stack_module_state(self.ensemble)

    def _build_model(self, device=None):
        return nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2, device=device),
            nn.LayerNorm(self.embed_dim * 2, device=device),
            nn.SiLU(),
            nn.Linear(self.embed_dim * 2, self.vocab_size, device=device)
        )

    def _ensemble_forward(self, params, buffers, x):
        return functional_call(self.meta_model, (params, buffers), (x,))

    def forward(self, x):
        x = rearrange(x, '(m bi) ... -> m bi ...', m=self.ensemble_size)
        x = torch.stack([m_i(x_i) for m_i, x_i in zip(self.ensemble, x)])
        # x = vmap(self._ensemble_forward)(self.ens_params, self.ens_buffers, x)
        x = rearrange(x, 'm bi ... -> (m bi) ...')
        return x

    def forward_all(self, x):
        # x = x.unsqueeze(0).expand(self.ensemble_size, *x.shape)
        # x = vmap(self._ensemble_forward)(self.ens_params, self.ens_buffers, x)
        # return x
        return torch.stack([m(x) for m in self.ensemble])

    def estimate_uncertainty(self, x):
        # TODO: implement a version without the for loop.
        # Use the "Jensen–Shannon divergence" as a variance measure between the discrete distributions:
        ensemble_outs = torch.stack([model_i(x) for model_i in self.ensemble])
        ensemble_dist = torch.distributions.Categorical(logits=ensemble_outs)
        ensemble_mean_dist = torch.distributions.Categorical(probs=torch.softmax(ensemble_outs, dim=-1).mean(0))
        jsd = ensemble_mean_dist.entropy() - ensemble_dist.entropy().mean(dim=0)

        probs = ensemble_mean_dist.probs

        # b = ensemble_outs.shape[1]
        # probs = torch.split(ensemble_dist.probs, b // self.ensemble_size, dim=1)
        # probs = torch.cat([p_i[i] for i, p_i in enumerate(probs)], dim=0)

        # ensemble_outs = torch.softmax(torch.stack([model_i(x) for model_i in self.ensemble]), dim=-1)
        # mean = ensemble_outs.mean(dim=0, keepdim=True)
        # var = torch.sum((ensemble_outs - mean) ** 2, dim=(0, -1)) / (self.ensemble_size - 1)
        # return var, mean.squeeze(0)

        return jsd, probs


class RewardHead(nn.Module):

    def __init__(self, embed_dim: int, num_values: int = 129, device=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reward_head_values = sym_exp(torch.linspace(-15, 15, num_values, device=device)) / 100
        self.model = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2, device=device),
            nn.SiLU(),
            # nn.Linear(embed_dim * 2, self.reward_head_values.numel())
        )
        self.head = HLGaussCategoricalRegressionHead(
            embed_dim * 2,
            num_values,
            support=self.reward_head_values,
            sym_log_normalize=True,
            device=device
        )
        # bias = -self.reward_head_values.abs() / (self.reward_head_values.max() / 5)
        # self.model[-1].bias.data = bias

    def forward(self, x):
        return self.head(self.model(x))


class POPWorldModel(nn.Module):
    def __init__(
            self, tokens_per_obs_dict: dict[ObsModality, int], obs_vocab_size: dict[ObsModality, int],
            action_encoder: ActionEncoder,
            retnet_cfg: RetNetConfig, context_length: int = 2,
            compute_states_parallel: bool = True, shared_embeddings: bool = True,
            shared_prediction_token: bool = False, obs_emb_dim: Optional[int] = None,
            enable_curiosity: bool = False, device=None, *args, **kwargs
    ) -> None:
        tokens_per_obs = sum(tokens_per_obs_dict.values())
        super().__init__()
        assert isinstance(tokens_per_obs, int)
        assert set(tokens_per_obs_dict.keys()) == set(obs_vocab_size.keys())

        self.tokens_per_obs = tokens_per_obs
        self.tokens_per_obs_dict = tokens_per_obs_dict
        self.tokens_per_action = action_encoder.action_sequence_length
        self.obs_vocab_size = obs_vocab_size
        self.config = retnet_cfg
        self._device = device
        self.enable_curiosity = enable_curiosity

        self._ordered_modalities = [
            modality for modality in ObsModality if modality in self.tokens_per_obs_dict
        ]

        self.head_observations = nn.ModuleDict(
            {k.name: self._build_obs_head(get_vocab_head_dim(vocab_size)) for k, vocab_size in obs_vocab_size.items()}
        )

        self.curiosity_head = nn.ModuleDict({k.name: EnsembleObsHead(
            ensemble_size=4, 
            embed_dim=retnet_cfg.embed_dim,
            vocab_size=get_vocab_head_dim(vocab_size),
            device=device
        ) for k, vocab_size in obs_vocab_size.items()}) if enable_curiosity else None

        self.head_rewards = self._build_reward_head()

        self.head_ends = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(retnet_cfg.embed_dim, retnet_cfg.embed_dim, device=device),
            nn.LayerNorm(retnet_cfg.embed_dim, device=device),
            nn.SiLU(),
            nn.Linear(retnet_cfg.embed_dim, 2, device=device)
        )

        self.context_length = context_length
        self.config = retnet_cfg

        self.shared_embeddings = shared_embeddings
        self.obs_embeddings = nn.ModuleDict({
            k.name: make_embeddings(vocab_size, retnet_cfg.embed_dim, device=device)
            for k, vocab_size in obs_vocab_size.items()
            if k != ObsModality.image or (not shared_embeddings)
        })

        self._model = self._build_model()
        self._model.out = nn.Identity()

        self.obs_emb_dim = obs_emb_dim
        self.obs_emb_map = self._build_emb_map()

        self.action_encoder = action_encoder

        self.placeholder_embeddings = nn.Embedding(self.tokens_per_block, retnet_cfg.embed_dim, device=device)
        self.compute_states_parallel = compute_states_parallel
        self.compute_states_parallel_inference = False
        self.pred_tokens_version = 'shared' if shared_prediction_token else 'per-token'

        logger.info(f"Initialized {self.__class__}. Shared embeddings: {self.shared_embeddings}")

    @property
    def tokens_per_block(self) -> int:
        return self.tokens_per_obs + self.tokens_per_action

    @property
    def ordered_modalities(self) -> list[ObsModality]:
        return self._ordered_modalities

    def _build_obs_head(self, obs_vocab_size: int) -> nn.Module:
        embed_dim = self.config.embed_dim
        device = self.device
        return nn.Sequential(
            # nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 2, device=device),
            nn.LayerNorm(embed_dim * 2, device=device),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, obs_vocab_size, device=device)
        )

    def _build_reward_head(self) -> nn.Module:
        return RewardHead(self.config.embed_dim, device=self.device)

    def _build_emb_map(self):
        if self.obs_emb_dim is None or self.obs_emb_dim == self.config.embed_dim:
            return None
        else:
            return nn.Linear(self.obs_emb_dim, self.config.embed_dim, device=self._device)

    def __repr__(self):
        return "world_model"

    def _build_model(self):
        decoder_layers = [
            POPRetNetDecoderLayer(
                self.config.embed_dim,
                self.config.num_heads,
                tokens_per_block=self.tokens_per_block,
                dropout=self.config.dropout,
                dim_feedforward=4 * self.config.embed_dim,
                mask_type=self.config.mask_type,
                decay_scale_min_num_blocks=self.config.decay_scale_min_num_blocks,
                decay_scale_max_num_blocks=self.config.decay_scale_max_num_blocks,
                device=self.device,
            ) for _ in range(self.config.num_layers)
        ]
        return POPRetNetDecoder(decoder_layers)

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def get_empty_state(self) -> RecurrentState:
        return RecurrentState(None, 0)

    @torch.no_grad()
    def get_obs_tokens(self, raw_obs: MultiModalObs, tokenizer: MultiModalTokenizer) -> dict[ObsModality, Tensor]:
        obs_tokens = {k: v.tokens for k, v in tokenizer.encode(raw_obs, should_preprocess=True).items()}

        for m, o in obs_tokens.items():
            if m == ObsModality.token_2d:
                assert o.dim() == 4, f"Got shape {o.shape}"
            else:
                assert o.dim() == 3, f"Got shape {o.shape}"
        return obs_tokens

    def embed_obs_tokens(self, obs_tokens: dict[ObsModality, Tensor], tokenizer: MultiModalTokenizer) -> Tensor:
        embs = []
        for modality in self.ordered_modalities:
            if modality == ObsModality.image and self.shared_embeddings:
                image_tokens = obs_tokens[modality]
                assert image_tokens.dim() == 3
                img_tokens_emb = tokenizer.tokenizers[ObsModality.image.name].to_codes(image_tokens.flatten(0, 1))
                img_tokens_emb = rearrange(img_tokens_emb, '(b t) e h w -> b t (h w) e', t=image_tokens.shape[1])
                if self.obs_emb_map is not None:
                    img_tokens_emb = self.obs_emb_map(img_tokens_emb)
                embs.append(img_tokens_emb)
            else:
                tokens_emb = self.obs_embeddings[modality.name](obs_tokens[modality])
                embs.append(tokens_emb)

        obs_tokens_emb = torch.cat(embs, dim=2)

        assert obs_tokens_emb.dim() == 4 and obs_tokens_emb.shape[2] == self.tokens_per_obs
        return obs_tokens_emb

    def embed_actions(self, actions: Tensor) -> Tensor:
        return self.action_encoder.embed_actions(actions)

    def get_tokens_emb(self, obs_tokens, actions, tokenizer: MultiModalTokenizer) -> Tensor:
        obs_tokens_emb = self.embed_obs_tokens(obs_tokens, tokenizer=tokenizer)
        assert obs_tokens_emb.dim() == 4, f"Got {obs_tokens_emb.dim()} ({obs_tokens_emb.shape})"
        actions_emb = self.embed_actions(actions)
        assert actions_emb.dim() == 4, f"Got {actions_emb.dim()} ({actions_emb.shape})"
        return torch.cat([obs_tokens_emb, actions_emb], dim=2)

    def sample_rewards_ends(self, outputs) -> tuple[Tensor, Tensor]:
        k1 = self.tokens_per_block
        if outputs.dim() == 3:  # (b (t k1) e)
            if outputs.shape[1] == k1 - self.tokens_per_action:
                relevant_latents = rearrange(outputs[:, -1], 'b e -> b 1 e')
            else:
                relevant_latents = rearrange(
                    outputs,
                    'b (t k1) e -> b t k1 e',
                    k1=k1
                )[:, :, -self.tokens_per_action - 1]
        else:
            assert outputs.dim() == 4  # (b t k1 e)
            relevant_latents = outputs[:, :, -self.tokens_per_action - 1]

        rewards = self.head_rewards(relevant_latents)
        ends_logits = self.head_ends(relevant_latents)
        ends = sample_categorical(logits=ends_logits).bool()

        return rewards, ends

    def forward(
            self,
            tokens_emb: torch.FloatTensor,
            recurrent_state: Optional[RecurrentState] = None
    ):
        assert tokens_emb.dim() == 4, f"Got {tokens_emb.dim()} instead of 4 ({tokens_emb.shape})"

        initial_state = recurrent_state
        if initial_state is None:
            initial_state = self.get_empty_state()

        assert isinstance(self._model, POPRetNetDecoder)

        if self.compute_states_parallel:
            return self._compute_train_forward_parallel(tokens_emb, initial_state)
        else:
            return self._compute_train_forward_sequential(tokens_emb, initial_state)

    @torch.no_grad()
    def forward_inference(self, tokens_emb: torch.FloatTensor, recurrent_state: Optional[RecurrentState] = None):
        assert isinstance(self._model, POPRetNetDecoder)

        assert tokens_emb.shape[1] < self.config.max_blocks
        tokens_emb = rearrange(tokens_emb, 'b t k1 e -> b (t k1) e')

        assert tokens_emb.shape[1] > 1, 'unsupported length!'

        if not self.compute_states_parallel_inference:
            outs, recurrent_state.state = self._model.forward_chunkwise(tokens_emb, recurrent_state.n,
                                                                        recurrent_state.state)
        else:
            raise NotImplementedError('Not yet implemented.')

        recurrent_state.n += tokens_emb.shape[1]
        return outs

    def _compute_train_forward_sequential(self, tokens_emb, initial_state: Optional[RecurrentState]):
        bsz, num_steps = tokens_emb.shape[:2]
        pred_tokens_emb = self._get_prediction_tokens_embeddings(bsz, 1, tokens_emb.device, obs_only=True)
        outputs = []

        for t in range(num_steps):
            pred_outs, _ = self._model.forward_chunkwise(pred_tokens_emb, initial_state.n, initial_state.state)
            outputs.append(pred_outs)

            tokens_emb_t = tokens_emb[:, t]
            step_outs, initial_state.state = self._model.forward_chunkwise(tokens_emb_t, initial_state.n, initial_state.state)

            initial_state.n += tokens_emb_t.shape[1]
            outputs.append(step_outs[:, -1:])
            assert tokens_emb_t.shape[1] == self.tokens_per_block

        return torch.cat(outputs, dim=1)

    def _compute_train_forward_parallel(self, tokens_emb, initial_state: Optional[RecurrentState]):
        bsz, num_steps = tokens_emb.shape[:2]
        n_action_tokens = self.tokens_per_action

        blocks_per_chunk = self.config.blocks_per_chunk
        n_chunks = ceil(num_steps / blocks_per_chunk)
        outs = []
        for i in range(n_chunks):
            start, stop = i * blocks_per_chunk, min((i+1) * blocks_per_chunk, num_steps)
            tokens_emb_i = tokens_emb[:, start:stop].flatten(1, 2)

            pred_tokens_emb = self._get_prediction_tokens_embeddings(bsz, stop - start, tokens_emb.device, obs_only=True)
            pred_tokens_emb = rearrange(pred_tokens_emb, 'b (t k) e -> b t k e', t=stop - start, k=self.tokens_per_obs)

            tokens_outs, pred_tokens_outs, state = self._model.pop_forward(
                tokens_emb_i,
                x_pred_tokens=pred_tokens_emb,
                start_idx=initial_state.n,
                prev_states=initial_state.state
            )

            # pred_tokens_outs = rearrange(pred_tokens_outs, 'b (t k1) d -> b t k1 d', k1=self.tokens_per_block)
            tokens_outs = rearrange(tokens_outs, 'b (t k1) d -> b t k1 d', k1=self.tokens_per_block)
            pred_tokens_outs = torch.cat([pred_tokens_outs, tokens_outs[:, :, -n_action_tokens:]], dim=2)
            assert pred_tokens_outs.shape[2] == self.tokens_per_block, f"got {pred_tokens_outs.shape[2]} instead of {self.tokens_per_block}"

            pred_tokens_outs = rearrange(pred_tokens_outs, 'b t k1 d -> b (t k1) d')
            outs.append(pred_tokens_outs)

            initial_state.state = state
            initial_state.n += tokens_emb_i.shape[1]
        return torch.cat(outs, dim=1)

    def compute_next_obs_pred_latents(self, recurrent_state: RecurrentState):
        assert recurrent_state is not None and isinstance(recurrent_state, RecurrentState)
        assert len(recurrent_state.state) > 0 and recurrent_state.state[0] is not None
        batch_size = recurrent_state.state[0].shape[0]
        device = recurrent_state.state[0].device
        pred_tokens_emb = self._get_prediction_tokens_embeddings(batch_size, 1, device, obs_only=True)
    
        return self._model.forward_chunkwise(pred_tokens_emb, recurrent_state.n, recurrent_state.state)

    def _get_prediction_tokens_embeddings(self, batch_size: int, num_steps: int, device, obs_only: bool = False):
        num_tokens = self.tokens_per_block if not obs_only else self.tokens_per_obs
        if self.pred_tokens_version == 'shared':
            return self.placeholder_embeddings(torch.zeros(batch_size, num_steps * num_tokens, device=device).long())
        elif self.pred_tokens_version == 'per-token':
            tokens = torch.arange(num_tokens, device=device)
            tokens = rearrange(tokens, 'k1 -> 1 1 k1').expand(batch_size, num_steps, -1).flatten(1, 2)
            return self.placeholder_embeddings(tokens)
        else:
            raise ValueError(f"Pred tokens version '{self.pred_tokens_version}' not supported.")
    
    def compute_loss(self, batch: Batch, tokenizer: MultiModalTokenizer, **kwargs: Any) -> tuple[LossWithIntermediateLosses, dict]:
        obs_tokens = self.get_obs_tokens(batch['observations'], tokenizer)
        tokens_emb = self.get_tokens_emb(obs_tokens, batch['actions'], tokenizer=tokenizer)

        outputs = self(tokens_emb)

        loss_obs, obs_per_sample_loss, curiosity_loss = self.get_next_token_logits_and_labels(outputs, obs_tokens, batch['mask_padding'])

        loss_rewards, loss_ends, reward_per_sample_loss = self.get_rewards_ends_losses(
            outputs,
            batch['mask_padding'],
            batch['rewards'],
            batch['ends']
        )

        info = {'per_sample_loss': obs_per_sample_loss + reward_per_sample_loss}
        losses = {
            'loss_obs': loss_obs,
            'loss_rewards': loss_rewards,
            'loss_ends': loss_ends
        }
        if curiosity_loss is not None:
            losses['curiosity_loss'] = curiosity_loss

        return LossWithIntermediateLosses(**losses), info

    def get_next_token_logits_and_labels(self, outputs: Tensor, obs_tokens: dict[ObsModality, Tensor], mask_padding: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pred_mask = mask_padding.clone()
        pred_mask[:, :self.context_length] = 0

        labels = [obs_tokens[m][pred_mask].flatten() for m in self.ordered_modalities]

        segment_lengths = [self.tokens_per_obs_dict[m] for m in self.ordered_modalities]
        segment_lengths.append(self.tokens_per_action)
        assert sum(segment_lengths) == self.tokens_per_block
        outputs = rearrange(outputs, "b (t k) e -> b t k e", k=self.tokens_per_block)
        segmented_outputs = torch.split(outputs, segment_lengths, dim=2)[:-1]  # discard the outputs of action tokens
        logits = [self.head_observations[m.name](o_m[pred_mask]).flatten(0, 1)
                  for m, o_m in zip(self.ordered_modalities, segmented_outputs)]

        loss_obs = torch.cat([
            modality_to_obs_loss[self.ordered_modalities[i]](
                logits[i],
                labels[i],
                self.obs_vocab_size[self.ordered_modalities[i]],
                reduction='none'
            ).reshape(-1, segment_lengths[i])
            for i in range(len(segment_lengths)-1)
        ], dim=1)

        if self.enable_curiosity:
            curiosity_logits = [self.curiosity_head[m.name](o_m.detach())[pred_mask].flatten(0, -2)
                                for m, o_m in zip(self.ordered_modalities, segmented_outputs)]
            curiosity_loss = torch.stack([
                modality_to_obs_loss[self.ordered_modalities[i]](
                    curiosity_logits[i],
                    labels[i],
                    self.obs_vocab_size[self.ordered_modalities[i]],
                    reduction='sum'
                )
                for i in range(len(segment_lengths)-1)
            ]).sum() / (self.tokens_per_obs * pred_mask.sum())
        else:
            curiosity_loss = None

        per_sample_counts = pred_mask.sum(dim=1).tolist()
        per_sample_loss = torch.split(loss_obs.detach(), per_sample_counts, dim=0)
        per_sample_loss = torch.stack([l_i.mean() for l_i in per_sample_loss]).flatten()

        return loss_obs.mean(), per_sample_loss, curiosity_loss

    def get_rewards_ends_losses(self, outputs: Tensor, mask_padding: Tensor, rewards, ends) -> tuple[
        Tensor, Tensor, Tensor]:
        relevant_elements_mask = torch.logical_and(mask_padding[:, :-1], mask_padding[:, 1:])
        relevant_labels_mask = F.pad(relevant_elements_mask, (0, 1), value=0)
        relevant_latents_mask = F.pad(relevant_elements_mask, (1, 0), value=0)

        # relevant_elements_mask[:, :self.context_length] = 0

        latents = rearrange(outputs, 'b (t k1) e -> b t k1 e', k1=self.tokens_per_block)[:, :,
                  -self.tokens_per_action - 1]
        # predicted_rewards = self.head_rewards(latents[torch.where(relevant_latents_mask)]).flatten()
        ends_logits = self.head_ends(latents.flatten(0, 1))

        mask_fill = torch.logical_not(relevant_labels_mask)
        ignore_value = -100
        rewards_labels = rewards[torch.where(relevant_labels_mask)]
        ends_labels = ends.masked_fill(mask_fill, ignore_value).flatten()

        reward_latents = self.head_rewards.model(latents[torch.where(relevant_latents_mask)])
        loss_rewards = self.head_rewards.head.compute_loss(reward_latents, rewards_labels, reduction='none')
        loss_ends = F.cross_entropy(ends_logits, ends_labels)

        per_sample_losses_count = relevant_latents_mask.flatten(1).sum(dim=1).tolist()
        per_sample_losses = torch.split(loss_rewards.detach(), per_sample_losses_count)
        per_sample_losses = torch.stack([l_i.mean() for l_i in per_sample_losses]).flatten()

        return loss_rewards.mean(), loss_ends, per_sample_losses

    @property
    def uses_pop(self) -> bool:
        return True

    def sample_obs_tokens(self, outputs) -> dict[ObsModality, Tensor]:
        assert outputs.shape[1] == self.tokens_per_obs
        outputs = torch.split(outputs, [self.tokens_per_obs_dict[m] for m in self.ordered_modalities], dim=1)
        modality_to_sampler = {
            m: sample_categorical for m in self.ordered_modalities if m != ObsModality.token_2d
        }
        if ObsModality.token_2d in self.ordered_modalities:
            modality_to_sampler[ObsModality.token_2d] = MultiCategoricalSampler(
                vocab_sizes=self.obs_vocab_size[ObsModality.token_2d]
            )
        tokens = {
            m: modality_to_sampler[m](logits=self.head_observations[m.name](outputs[i]))
            for i, m in enumerate(self.ordered_modalities)
        }
        return tokens


class DiscretePOPWorldModel(POPWorldModel):

    def embed_actions(self, actions: Tensor) -> Tensor:
        return self.action_embeddings(rearrange(actions.long(), 'b l -> b l 1'))

    # def _build_reward_head(self) -> nn.Module:
    #     return nn.Sequential(
    #         # nn.ReLU(),
    #         nn.Linear(self.config.embed_dim, self.config.embed_dim, device=self.device),
    #         nn.SiLU(),
    #         nn.Linear(self.config.embed_dim, 3, device=self.device)
    #     )
    #
    # def get_rewards_ends_losses(self, outputs: Tensor, mask_padding: Tensor, rewards, ends) -> tuple[Tensor, Tensor]:
    #     relevant_elements_mask = mask_padding.clone()
    #     relevant_elements_mask[:, :self.context_length] = 0
    #
    #     action_tokens_logits = rearrange(outputs, 'b (t k1) e -> b t k1 e', k1=self.tokens_per_block)[:, :, -1]
    #     rewards_logits = self.head_rewards(action_tokens_logits.flatten(0, 1))
    #     ends_logits = self.head_ends(action_tokens_logits.flatten(0, 1))
    #
    #     mask_fill = torch.logical_not(relevant_elements_mask)
    #     ignore_value = -100
    #     rewards_labels = (rewards.sign() + 1).masked_fill(mask_fill, ignore_value).long().reshape(-1)
    #     ends_labels = ends.masked_fill(mask_fill, ignore_value).reshape(-1)
    #
    #     loss_rewards = F.cross_entropy(rewards_logits, rewards_labels)
    #     loss_ends = F.cross_entropy(ends_logits, ends_labels)
    #
    #     return loss_rewards, loss_ends
    #
    # def sample_rewards_ends(self, outputs) -> tuple[Tensor, Tensor]:
    #     k1 = self.tokens_per_block
    #     if outputs.dim() == 3:  # (b (t k1) e)
    #         outputs = rearrange(outputs, 'b (t k1) e -> b t k1 e', k1=k1)
    #
    #     assert outputs.shape[2] == k1
    #     relevant_latents = outputs[:, :, -1]
    #
    #     rewards_logits = self.world_model.head_rewards(relevant_latents)
    #     rewards = sample_categorical(logits=rewards_logits).float() - 1
    #     ends_logits = self.head_ends(relevant_latents)
    #     ends = sample_categorical(logits=ends_logits).bool()
    #
    #     return rewards, ends


class ContinuousPOPWorldModel(POPWorldModel):

    def __init__(
            self, tokens_per_obs_dict: dict[ObsModality, int], obs_vocab_size: dict[ObsModality, int],
            action_dim: int, action_vocab_size: int,
            retnet_cfg: RetNetConfig, context_length: int = 2, compute_states_parallel: bool = True,
            shared_embeddings: bool = True, obs_emb_dim: Optional[int] = None, shared_prediction_token: bool = False,
            enable_curiosity: bool = False, device=None, tokenize_actions: bool = False, **kwargs
    ):
        self.tokenize_actions = tokenize_actions
        action_seq_len = action_dim if tokenize_actions else 1
        super().__init__(
            tokens_per_obs_dict=tokens_per_obs_dict, obs_vocab_size=obs_vocab_size,
            action_vocab_size=action_vocab_size, retnet_cfg=retnet_cfg,
            context_length=context_length, tokens_per_action=action_seq_len,
            compute_states_parallel=compute_states_parallel, shared_embeddings=shared_embeddings,
            obs_emb_dim=obs_emb_dim, shared_prediction_token=shared_prediction_token,
            enable_curiosity=enable_curiosity, device=device, **kwargs
        )
        self.action_dim = action_dim
        self.action_projection = nn.Linear(action_dim, retnet_cfg.embed_dim, device=device)
        self.action_tokenizer = HardCodedVectorTokenizer(
            input_dim=action_dim, vector_quantizer=UniformVQ(vmin=-1, vmax=1, support_size=action_vocab_size),
            device=device
        )

    def embed_actions(self, actions: Tensor) -> Tensor:
        assert actions.dim() == 3, f"Got shape {actions.shape}"
        if not self.tokenize_actions:
            return rearrange(self.action_projection(actions), 'b t e -> b t 1 e')
        else:
            action_tokens = self.action_tokenizer.encode(actions).tokens  # b t d
            return self.action_embeddings(action_tokens)  # b t d e





