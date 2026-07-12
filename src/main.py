import faulthandler
faulthandler.enable()
import os
import multiprocessing
from collections import defaultdict
from functools import partial
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium.spaces
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_cudnn_sdp(False)
# torch.backends.cudnn.allow_tf32 = False
# from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
# allow_ops_in_compiled_graph()
from tqdm import tqdm
import wandb
from loguru import logger

from async_evaluator import AsyncEvaluator, ConsoleEvent, EvalResult
from agent import Agent
from collector import Collector
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from make_reconstructions import make_reconstructions_from_batch
from models.actor_critic import (
    DiscreteActorCriticLS, DContinuousActorCriticLS,
    ContinuousActorCriticLS, ActorCriticLS, ImageLatentObsEncoder, VectorObsEncoder, TokenObsEncoder,
    MultiDiscreteActorCriticLS
)
from models.embedding import DiscreteActionEncoder, MultiDiscreteActionEncoder, ContinuousActionEncoder
from models.world_model import POPWorldModel
from models.tokenizer import MultiModalTokenizer, DummyTokenizer
from utils import (
    configure_optimizer, set_seed, GradNormInfo, TokenizerInfoHandler,
    TrainerInfoHandler, ControllerInfoHandler, ObsModality,
    TrainingDisplay, print_collector_log,
)
from dataset import get_dataloader, CuriousReplayDistribution, EpisodeDirManager, NpCuriousReplayDistribution


def get_driver_version():
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "N/A"


Env = Union[SingleProcessEnv, MultiProcessEnv]


def create_env(cfg_env: DictConfig, num_envs: int) -> Env:
    env_fn = partial(instantiate, config=cfg_env)
    return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)


def create_run_envs(cfg: DictConfig) -> Tuple[Optional[Env], Optional[Env]]:
    train_env = create_env(cfg.env.train, cfg.collection.train.num_envs) if cfg.training.should else None
    async_eval = cfg.evaluation.should and cfg.training.should and cfg.evaluation.async_eval.enabled
    test_env = create_env(cfg.env.test, cfg.collection.test.num_envs) if cfg.evaluation.should and not async_eval else None
    return train_env, test_env


def close_run_envs(train_env: Optional[Env], test_env: Optional[Env]) -> None:
    if train_env is not None:
        train_env.close()
    if test_env is not None and test_env is not train_env:
        test_env.close()


class RunMetadata:
    def __init__(
            self,
            epoch: int = 1,
            best_eval_score: Optional[float] = None,
            epoch_of_best_score: Optional[int] = None
    ):
        self.epoch = epoch
        self.best_eval_score = best_eval_score
        self.epoch_of_best_score = epoch_of_best_score

    def update_eval_score(self, score: float, epoch: Optional[int] = None):
        if self.best_eval_score is None or score >= self.best_eval_score:
            self.best_eval_score = score
            self.epoch_of_best_score = self.epoch if epoch is None else epoch

    @property
    def is_current_epoch_best(self):
        return self.epoch_of_best_score == self.epoch

    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'best_eval_score': self.best_eval_score,
            'epoch_of_best_score': self.epoch_of_best_score,
        }


def build_agent(env, cfg, device):
    try:
        project_root = Path(hydra.utils.get_original_cwd())
    except ValueError:
        project_root = Path(os.environ.get('SIMULUS_ORIGINAL_CWD', Path.cwd()))
    is_continuous_env = isinstance(env.action_space, gymnasium.spaces.Box)

    tokenizers = {}
    ac_encoders = {}
    if hasattr(cfg.tokenizer, 'image'):
        vgg_lpips_rel_path = cfg.tokenizer.image.vgg_lpips_ckpt_path
        cfg.tokenizer.image.vgg_lpips_ckpt_path = (project_root / vgg_lpips_rel_path).absolute()
        tokenizers[ObsModality.image] = instantiate(cfg.tokenizer.image)
        # tokenizers[ObsModality.image].compile()

        ac_encoders[ObsModality.image] = ImageLatentObsEncoder(
            tokens_per_obs=tokenizers[ObsModality.image].tokens_per_obs,
            embed_dim=cfg.tokenizer.image.embed_dim,
            num_layers=cfg.actor_critic.num_layers,
            device=device,
        )

    if hasattr(cfg.tokenizer, 'vector'):
        obs_dim = env.observation_space[ObsModality.vector].shape[0]
        tokenizers[ObsModality.vector] = instantiate(cfg.tokenizer.vector, input_dim=obs_dim)

        ac_encoders[ObsModality.vector] = VectorObsEncoder(
            obs_dim=obs_dim,
            device=device,
        )

    if ObsModality.token in env.modalities:
        nvec = env.observation_space[ObsModality.token].nvec
        tokenizers[ObsModality.token] = DummyTokenizer(nvec=nvec)

        ac_encoders[ObsModality.token] = TokenObsEncoder(
            nvec=nvec,
            embed_dim=cfg.actor_critic.tokens_embed_dim,
            device=device
        )

    if ObsModality.token_2d in env.modalities:
        nvec = env.observation_space[ObsModality.token_2d].nvec
        tokenizers[ObsModality.token_2d] = DummyTokenizer(nvec=nvec)

        ac_encoders[ObsModality.token_2d] = TokenObsEncoder(
            nvec=nvec,
            embed_dim=cfg.actor_critic.tokens_embed_dim,
            device=device
        )

    assert set(
        tokenizers.keys()) == env.modalities, f"Modalities mismatch: env: {env.modalities}, tokenizers: {tokenizers.keys()}"
    tokenizer = MultiModalTokenizer(tokenizers=tokenizers)

    # Resolve backbone config — either 'transformer' or 'retnet' sub-config.
    if hasattr(cfg.world_model, 'transformer') and cfg.world_model.transformer is not None:
        backbone_cfg = instantiate(cfg.world_model.transformer)
        embed_dim = cfg.world_model.transformer.embed_dim
    else:
        backbone_cfg = instantiate(cfg.world_model.retnet)
        embed_dim = cfg.world_model.retnet.embed_dim

    # Init world model + controller:
    if is_continuous_env:
        assert len(env.action_space.shape) == 1
        action_dim = env.action_space.shape[0]
        action_encoder = ContinuousActionEncoder(
            action_dim=action_dim,
            action_vocab_size=cfg.actor_critic.n_action_quant_levels,
            embed_dim=embed_dim,
            device=device,
            tokenize_actions=cfg.world_model.tokenize_actions,
        )
        actor_critic = DContinuousActorCriticLS(
            **cfg.actor_critic,
            action_dim=action_dim,
            obs_encoders=ac_encoders,
            context_len=cfg.world_model.context_length,
            device=device,
        )
    elif isinstance(env.action_space, gymnasium.spaces.MultiDiscrete):
        action_encoder = MultiDiscreteActionEncoder(
            nvec=env.action_space.nvec,
            embed_dim=embed_dim,
            device=device,
        )
        actor_critic = MultiDiscreteActorCriticLS(
            actions_nvec=env.action_space.nvec,
            obs_encoders=ac_encoders,
            context_len=cfg.world_model.context_length,
            device=device,
            **cfg.actor_critic,
        )
    else:
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        action_encoder = DiscreteActionEncoder(
            num_actions=env.num_actions,
            embed_dim=embed_dim,
            device=device,
        )
        actor_critic = DiscreteActorCriticLS(
            **cfg.actor_critic,
            act_vocab_size=env.num_actions,
            obs_encoders=ac_encoders,
            context_len=cfg.world_model.context_length,
            device=device
        )

    # Strip backbone sub-configs from kwargs — only the instantiated backbone_cfg is passed.
    wm_extra_cfg = {k: v for k, v in cfg.world_model.items() if k not in ('retnet', 'transformer')}
    world_model = POPWorldModel(
        tokens_per_obs_dict=tokenizer.tokens_per_obs_dict,
        obs_vocab_size=tokenizer.vocab_size,
        action_encoder=action_encoder,
        backbone_cfg=backbone_cfg,
        device=device,
        **wm_extra_cfg
    )
    # world_model.compile()
    # actor_critic.compile()

    return Agent(tokenizer, world_model, actor_critic, use_bf16_autocast=cfg.common.use_bf16_autocast)


class Trainer:
    def __init__(self, cfg: DictConfig, train_env: Optional[Env], test_env: Optional[Env]) -> None:
        self.cfg = cfg
        self.train_env = train_env
        self.test_env = test_env
        assert self.train_env is not None or self.test_env is not None

        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_config["system"] = {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "N/A",
            "gpu_count": torch.cuda.device_count(),
            "gpu_driver_version": get_driver_version(),
            "gpu_memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1e9,
                2
                ) if torch.cuda.is_available() else 0,
        }

        wandb.init(
            config=wandb_config,
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        # torch.autograd.set_detect_anomaly(True)
        torch.set_printoptions(profile='short', sci_mode=False)
        torch.set_float32_matmul_precision(cfg.common.float32_matmul_precision)
        # "highest" keeps full fp32 matmul/conv behavior; "high"/"medium" allow TF32 for faster fp32 CUDA kernels.
        torch.backends.cuda.matmul.allow_tf32 = cfg.common.float32_matmul_precision != 'highest'
        torch.backends.cudnn.allow_tf32 = cfg.common.float32_matmul_precision != 'highest'

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        project_root = Path(hydra.utils.get_original_cwd())
        os.environ['SIMULUS_ORIGINAL_CWD'] = str(project_root)
        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / f'base.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            if not cfg.common.metrics_only_mode:
                shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
                shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.episode_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)

        disable_saving = cfg.common.metrics_only_mode
        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=cfg.collection.train.num_episodes_to_save, disable_saving=disable_saving)
        self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination', max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save, disable_saving=disable_saving)

        if self.cfg.training.should:
            self.train_dataset = instantiate(cfg.datasets.train)
            self.train_collector = Collector(self.train_env, self.train_dataset, episode_manager_train)

        if self.cfg.evaluation.should and not self.is_async_eval_enabled:
            episode_manager_test = EpisodeDirManager(self.episode_dir / 'test', max_num_episodes=cfg.collection.test.num_episodes_to_save, disable_saving=disable_saving)
            self.test_dataset = instantiate(cfg.datasets.test)
            self.test_collector = Collector(self.test_env, self.test_dataset, episode_manager_test)

        assert (self.cfg.training.should and self.train_env is not None) or (self.cfg.evaluation.should and self.test_env is not None)
        # logger.info(f"Obs space: {env.observation_space}")
        # logger.info(f"Action space size: {env.num_actions}")

        self.agent = build_agent(
            env=self.train_env if self.cfg.training.should else self.test_env,
            cfg=self.cfg,
            device=self.device,
        )

        if self.agent.tokenizer.is_trainable:
            logger.info(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
            self.optimizer_tokenizer = torch.optim.AdamW(self.agent.tokenizer.parameters(),
                                                         lr=cfg.training.tokenizer.learning_rate)
            if ObsModality.image.name in self.agent.tokenizer.tokenizers:
                self.tokenizer_info_handler = TokenizerInfoHandler(
                    codebook_size=self.agent.tokenizer.tokenizers[ObsModality.image.name].vocab_size)

        logger.info(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
        logger.info(f'{sum(p.numel() for p in self.agent.actor_critic.parameters())} parameters in agent.actor_critic')

        self.optimizer_world_model = configure_optimizer(self.agent.world_model, cfg.training.world_model.learning_rate, cfg.training.world_model.weight_decay)
        # self.optimizer_world_model = torch.optim.AdamW(self.agent.world_model.parameters(), lr=cfg.training.world_model.learning_rate, weight_decay=cfg.training.world_model.weight_decay)
        self.optimizer_actor_critic = torch.optim.AdamW(self.agent.actor_critic.parameters(), lr=cfg.training.actor_critic.learning_rate)

        self.actor_critic_info_handler = ControllerInfoHandler()

        uniform_fraction = cfg.training.world_model.replay_sampling_uniform_fraction
        if uniform_fraction < 1.0:
            self.wm_crd = NpCuriousReplayDistribution(uniform_portion=uniform_fraction)
        else:
            self.wm_crd = None

        if cfg.initialization.agent.path_to_checkpoint is not None:
            self.agent.load(**cfg.initialization.agent, device=self.device)

        if cfg.initialization.dataset.path is not None and not cfg.common.resume:
            dataset_path = project_root / Path(cfg.initialization.dataset.path)
            logger.info(f"Loading experience data from path='{dataset_path.absolute()}'")
            self.train_dataset.load_disk_checkpoint(dataset_path)

        self.run_metadata = RunMetadata()
        self.display = TrainingDisplay(mode=cfg.common.display_mode)

        if cfg.common.resume:
            self.load_checkpoint()

        self.async_evaluator = None
        if self.is_async_eval_enabled:
            self.async_evaluator = AsyncEvaluator(
                cfg=self.cfg,
                ckpt_dir=self.ckpt_dir,
                media_dir=self.media_dir,
                best_eval_score=self.run_metadata.best_eval_score,
                epoch_of_best_score=self.run_metadata.epoch_of_best_score,
                save_snapshot=self.save_eval_snapshot,
            )

    @property
    def is_async_eval_enabled(self) -> bool:
        return self.cfg.evaluation.should and self.cfg.training.should and self.cfg.evaluation.async_eval.enabled

    def run(self) -> None:
        with self.display:
            for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):
                self.handle_async_console_events(self.async_evaluator.poll_console() if self.async_evaluator is not None else [])
                self.handle_async_eval_results(self.async_evaluator.poll() if self.async_evaluator is not None else [])
                self.run_metadata.epoch = epoch
                self.display.start_epoch(epoch, self.cfg.common.epochs)
                start_time = time.time()
                to_log = []

                if self.cfg.training.should:
                    if epoch <= self.cfg.collection.train.stop_after_epochs:
                        self.agent.eval()
                        collector_log = self.train_collector.collect(self.agent, epoch, **self.cfg.collection.train.config, display=self.display)
                        to_log += collector_log
                        self.display.show_collector(collector_log)
                    else:
                        if self.agent.world_model.enable_curiosity:
                            self.agent.world_model.enable_curiosity = False
                to_log += self.train_agent(epoch)

                if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0) and (epoch >= self.cfg.training.tokenizer.start_after_epochs):
                    if self.is_async_eval_enabled:
                        assert self.async_evaluator is not None
                        self.handle_async_console_events(self.async_evaluator.poll_console())
                        self.handle_async_eval_results(self.async_evaluator.poll())
                        self.async_evaluator.dispatch(epoch)
                    else:
                        assert self.test_dataset is not None and self.test_collector is not None
                        self.test_dataset.clear()
                        self.test_collector.reset()

                        collection_kwargs = {**self.cfg.collection.test.config}
                        kw_to_del = 'num_episodes_end' if 'num_episodes' in collection_kwargs else 'num_steps_end'
                        del collection_kwargs[kw_to_del]

                        test_collect_log = self.test_collector.collect(self.agent, epoch, **collection_kwargs, display=self.display)
                        to_log += test_collect_log
                        self.display.show_collector(test_collect_log)

                        self.run_metadata.update_eval_score(test_collect_log[-1]['test_dataset/return'], epoch=epoch)
                        self.display.update_info(f"Best epoch {self.run_metadata.epoch_of_best_score}, best score: {self.run_metadata.best_eval_score}")

                        eval_log = self.eval_agent(epoch)
                        to_log += eval_log

                skip_checkpoint = self.cfg.common.metrics_only_mode or (not self.cfg.common.do_checkpoint)
                if self.cfg.training.should and (epoch % self.cfg.evaluation.every == 0) and (not skip_checkpoint):
                    self.save_checkpoint(save_agent_only=False)

                to_log.append({'duration': (time.time() - start_time)})
                self.display.end_epoch(status=f"done in {to_log[-1]['duration']:.1f}s")
                for metrics in to_log:
                    wandb.log({'epoch': epoch, **metrics})

            if self.async_evaluator is not None:
                self.handle_async_eval_results(self.async_evaluator.wait_for_pending())
                self.handle_async_console_events(self.async_evaluator.poll_console())

        self.final_eval()
        self.finish()

    def handle_async_console_events(self, events: List[ConsoleEvent]) -> None:
        for event in events:
            style = 'yellow' if event.stream == 'stderr' else 'dim'
            self.display.console.print(event.text, end='', markup=False, style=style)

    def handle_async_eval_results(self, results: List[EvalResult]) -> None:
        for result in results:
            if result.error is not None:
                raise RuntimeError(f"Async evaluator failed for epoch {result.epoch}:\n{result.error}")

            self.run_metadata.best_eval_score = result.best_eval_score
            self.run_metadata.epoch_of_best_score = result.epoch_of_best_score
            if result.best_eval_score is not None:
                self.display.update_info(f"Best epoch {result.epoch_of_best_score}, best score: {result.best_eval_score}")

            if result.final:
                print_collector_log(result.metrics)
            else:
                collector_logs = [metrics for metrics in result.metrics if any(k.startswith('test_dataset/') for k in metrics)]
                eval_metrics = {
                    k: v
                    for metrics in result.metrics
                    for k, v in metrics.items()
                    if '/eval/' in k
                }
                if collector_logs:
                    collector_display_logs = [
                        {
                            **({'test_dataset/eval_epoch': result.epoch} if any('#' in k for k in metrics) else {}),
                            **metrics,
                        }
                        for metrics in collector_logs
                    ]
                    self.display.show_collector(collector_display_logs)
                if eval_metrics:
                    self.display.update_metrics(eval_metrics)

            for metrics in result.metrics:
                wandb.log({'epoch': result.epoch, **metrics})

            if (not self.cfg.common.metrics_only_mode) and self.cfg.common.do_checkpoint:
                torch.save(self.run_metadata.to_dict(), self.ckpt_dir / 'run_metadata.pt')

    def save_eval_snapshot(self, epoch: int) -> Path:
        snapshot_dir = self.ckpt_dir / 'eval_snapshots'
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot_path = snapshot_dir / f'epoch_{epoch:08d}.pt'
        tmp_path = snapshot_path.with_suffix('.tmp')
        torch.save(
            {k: v.detach().cpu() for k, v in self.agent.state_dict().items()},
            tmp_path,
        )
        os.replace(tmp_path, snapshot_path)
        return snapshot_path

    def final_eval(self):
        if self.is_async_eval_enabled:
            assert self.async_evaluator is not None
            self.handle_async_console_events(self.async_evaluator.poll_console())
            self.handle_async_eval_results(self.async_evaluator.wait_for_pending())
            self.handle_async_console_events(self.async_evaluator.poll_console())
            self.async_evaluator.dispatch(self.run_metadata.epoch + 1, final=True)
            self.handle_async_eval_results(self.async_evaluator.wait_for_pending())
            self.handle_async_console_events(self.async_evaluator.poll_console())
            return

        collection_kwargs = {**self.cfg.collection.test.config}
        epoch = self.run_metadata.epoch + 1

        if 'num_episodes' in collection_kwargs:
            collection_kwargs['num_episodes'] = collection_kwargs['num_episodes_end']
            kw_to_del = 'num_episodes_end'
        else:
            assert 'num_steps' in collection_kwargs
            collection_kwargs['num_steps'] = collection_kwargs['num_steps_end']
            kw_to_del = 'num_steps_end'

        del collection_kwargs[kw_to_del]

        test_collect_log = self.test_collector.collect(self.agent, epoch, **collection_kwargs)

        print_collector_log(test_collect_log)
        for metrics in test_collect_log:
            wandb.log({'epoch': epoch, **metrics})

    def train_agent(self, epoch: int) -> None:
        self.agent.train()
        self.agent.zero_grad()

        metrics_tokenizer, metrics_world_model, metrics_actor_critic = {}, {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
        cfg_actor_critic = self.cfg.training.actor_critic

        if self.agent.tokenizer is not None and self.agent.tokenizer.is_trainable:
            if epoch > cfg_tokenizer.start_after_epochs:
                metrics_tokenizer = self.train_component(
                    epoch,
                    self.agent.tokenizer,
                    self.optimizer_tokenizer,
                    sequence_length=1,
                    sample_from_start=True,
                    context_len=0,
                    info_handler=self.tokenizer_info_handler,
                    **cfg_tokenizer
                )
                self.display.update_metrics(metrics_tokenizer)
            self.agent.tokenizer.eval()

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.train_component(
                epoch,
                self.agent.world_model,
                self.optimizer_world_model,
                sequence_length=self.cfg.common.sequence_length,
                sample_from_start=True,
                tokenizer=self.agent.tokenizer,
                context_len=self.cfg.world_model.context_length,
                replay_dist=self.wm_crd,
                **cfg_world_model
            )
            self.display.update_metrics(metrics_world_model)
        self.agent.world_model.eval()

        critic_start = cfg_actor_critic.start_after_epochs
        actor_start = critic_start + self.cfg.training.actor_critic.critic_warmup_epochs
        if epoch > cfg_actor_critic.start_after_epochs:
            metrics_actor_critic = self.train_component(
                epoch, 
                self.agent.actor_critic, 
                self.optimizer_actor_critic,
                sequence_length=self.cfg.training.actor_critic.burn_in + self.cfg.world_model.context_length,
                sample_from_start=False,
                tokenizer=self.agent.tokenizer, 
                world_model=self.agent.world_model, 
                context_len=self.cfg.world_model.context_length + 1,  # we drop the last obs in case it's terminal.
                actor_start_epoch=actor_start,
                info_handler=self.actor_critic_info_handler,
                **cfg_actor_critic
            )
            self.display.update_metrics(metrics_actor_critic)
        self.agent.actor_critic.eval()

        return [{**metrics_tokenizer, **metrics_world_model, **metrics_actor_critic}]

    def train_component(
            self, epoch: int, component: nn.Module, optimizer: torch.optim.Optimizer, steps_per_epoch: int,
            batch_num_samples: int, grad_acc_steps: int, max_grad_norm: Optional[float],
            sequence_length: int, sample_from_start: bool, context_len: int, info_handler: TrainerInfoHandler = None,
            replay_dist: Optional[NpCuriousReplayDistribution] = None, **kwargs_loss: Any
    ) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        grad_norms_info = GradNormInfo()

        if info_handler is not None:
            info_handler.signal_epoch_start()

        dataloader = get_dataloader(
            self.train_dataset,
            context_len,
            sequence_length,
            batch_num_samples,
            shuffle=True,
            padding_strategy='right' if sample_from_start else 'left',
            obs_modalities=self.agent.tokenizer.modalities,
            replay_dist=replay_dist,
            num_workers=self.cfg.common.num_dataloader_workers,
        )

        data_iter = iter(dataloader)

        def _fetch_batch():
            nonlocal data_iter
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            return batch

        def _checked_to_device(batch):
            assert (batch['mask_padding'].sum(dim=1) > context_len).all()
            return self._to_device(batch)

        task = self.display.add_task(f"Training {component}", total=steps_per_epoch)
        for i in range(steps_per_epoch):
            optimizer.zero_grad()
            step_loss = 0.0
            for _ in range(grad_acc_steps):
                batch = _checked_to_device(_fetch_batch())

                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.cfg.common.use_bf16_autocast):
                    losses, info = component.compute_loss(batch, epoch=epoch, num_epochs=self.cfg.common.epochs, **kwargs_loss)

                if replay_dist is not None:
                    assert 'per_sample_loss' in info
                    replay_dist.update_losses(info['per_sample_loss'].detach().cpu().numpy())

                losses = losses / grad_acc_steps
                loss_total_step = losses.loss_total
                loss_total_step.backward()
                torch.cuda.synchronize()
                step_loss += loss_total_step.item()
                loss_total_epoch += loss_total_step.item() / steps_per_epoch

                if info_handler is not None:
                    info_handler.update_with_step_info(info)

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value / steps_per_epoch

            grad_norm = torch.nn.utils.clip_grad_norm_(
                component.parameters(), max_grad_norm if max_grad_norm is not None else float('inf')
            )
            grad_norms_info(grad_norm)

            if torch.isfinite(grad_norm):
                optimizer.step()
            torch.cuda.synchronize()
            self.display.update_task(task, advance=1, metrics=f"loss={step_loss:.4f}  grad_norm={grad_norm.item():.3f}")

        epoch_info = {}
        if info_handler is not None:
            epoch_info = {f'{str(component)}/train/{k}': v for k, v in info_handler.get_epoch_info().items()}
        for k, v in grad_norms_info.get_info().items():
            epoch_info[f"{str(component)}/train/{k}"] = v
        metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses, **epoch_info}
        return metrics

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        self.agent.eval()

        metrics_tokenizer, metrics_world_model = {}, {}

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model
        cfg_actor_critic = self.cfg.evaluation.actor_critic

        if epoch > cfg_tokenizer.start_after_epochs and self.agent.tokenizer is not None and self.agent.tokenizer.is_trainable:
            metrics_tokenizer = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples, sequence_length=1, context_length=0)

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.eval_component(
                self.agent.world_model, 
                cfg_world_model.batch_num_samples, 
                sequence_length=self.cfg.common.sequence_length, 
                context_length=self.cfg.world_model.context_length, 
                tokenizer=self.agent.tokenizer
            )

        # if epoch > cfg_actor_critic.start_after_epochs:
        #     self.inspect_imagination(epoch)

        if cfg_tokenizer.save_reconstructions and (not self.cfg.common.metrics_only_mode) and (self.agent.tokenizer is not None) and (ObsModality.image in self.agent.tokenizer.modalities):
            dataloader = get_dataloader(
                self.test_dataset,
                1,
                self.cfg.common.sequence_length,
                batch_size=3,
                shuffle=True,
                padding_strategy='right',
                obs_modalities=self.agent.tokenizer.modalities
            )
            batch = self._to_device(next(iter(dataloader)))
            make_reconstructions_from_batch(batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agent.tokenizer)

        return [metrics_tokenizer, metrics_world_model]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, batch_num_samples: int, sequence_length: int, context_length: int, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        steps = 0
        dataloader = get_dataloader(
                self.test_dataset,
                context_length,
                sequence_length,
                batch_num_samples,
                shuffle=True,
                padding_strategy='right',
                obs_modalities=self.agent.tokenizer.modalities
            )
        num_batches = int(np.ceil(len(dataloader) / sequence_length)) if len(dataloader) > sequence_length else len(dataloader)
        data_iter = iter(dataloader)
        task = self.display.add_task(f"Evaluating {component}", total=num_batches)
        for batch_i in range(num_batches):
            batch = next(data_iter)
            assert (batch['mask_padding'].sum(dim=1) > context_length).all()
            batch = self._to_device(batch)

            losses, info = component.compute_loss(batch, **kwargs_loss)
            loss_total_epoch += losses.loss_total.item()

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

            steps += 1
            self.display.update_task(task, advance=1)

        if steps == 0:
            return {}
        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f'{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def inspect_imagination(self, epoch: int) -> None:
        mode_str = 'imagination'
        # batch = self.test_dataset.sample_batch(batch_num_samples=self.episode_manager_imagination.max_num_episodes, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False)
        dataloader = get_dataloader(
                self.test_dataset,
                self.agent.world_model.context_length,
                1 + self.cfg.training.actor_critic.burn_in,
                batch_size=self.episode_manager_imagination.max_num_episodes,
                shuffle=True,
                padding_strategy='left',
                obs_modalities=self.agent.tokenizer.modalities
            )
        batch = next(iter(dataloader))
        outputs = self.agent.actor_critic.imagine(self._to_device(batch), self.agent.tokenizer, self.agent.world_model, horizon=self.cfg.evaluation.actor_critic.horizon, show_pbar=True)

        if isinstance(self.agent.actor_critic, ActorCriticLS):
            outputs.observations = torch.clamp(self.agent.tokenizer.tokenizers[ObsModality.image].decode(outputs.observations, should_postprocess=True), 0, 1).mul(255).byte()

        to_log = []
        for i, (o, a, r, d) in enumerate(zip(outputs.observations.cpu(), outputs.actions.cpu(), outputs.rewards.cpu(), outputs.ends.long().cpu())):  # Make everything (N, T, ...) instead of (T, N, ...)
            episode = Episode(o, a, r, d, torch.ones_like(d))
            episode_id = (epoch - 1 - self.cfg.training.actor_critic.start_after_epochs) * outputs.observations.size(0) + i
            if not self.cfg.common.metrics_only_mode:
                self.episode_manager_imagination.save(episode, episode_id, epoch)

            metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
            metrics_episode['episode_num'] = episode_id
            metrics_episode['action_histogram'] = wandb.Histogram(episode.actions.numpy(), num_bins=self.agent.world_model.act_vocab_size)
            to_log.append({f'{mode_str}/{k}': v for k, v in metrics_episode.items()})

        return to_log

    def _save_checkpoint(self, save_agent_only: bool) -> None:
        torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')
        if (not self.is_async_eval_enabled) and self.run_metadata.is_current_epoch_best:
            torch.save(self.agent.state_dict(), self.ckpt_dir / 'best.pt')
        if not save_agent_only:
            torch.save(self.run_metadata.to_dict(), self.ckpt_dir / 'run_metadata.pt')
            optimizers_states_dict = {
                "optimizer_world_model": self.optimizer_world_model.state_dict(),
                "optimizer_actor_critic": self.optimizer_actor_critic.state_dict(),
            }
            if self.agent.tokenizer is not None and self.agent.tokenizer.is_trainable:
                optimizers_states_dict["optimizer_tokenizer"] = self.optimizer_tokenizer.state_dict()
            torch.save(optimizers_states_dict, self.ckpt_dir / 'optimizer.pt')
            ckpt_dataset_dir = self.ckpt_dir / 'dataset'
            ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
            self.train_dataset.update_disk_checkpoint(ckpt_dataset_dir)
            if self.cfg.evaluation.should and not self.is_async_eval_enabled:
                if self.cfg.collection.test.store_dataset:
                    test_dataset_dir = self.ckpt_dir / 'test_dataset'
                    test_dataset_dir.mkdir(exist_ok=True, parents=False)
                    self.test_dataset.update_disk_checkpoint(test_dataset_dir)
                torch.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

    def save_checkpoint(self, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(
            src=self.ckpt_dir,
            dst=tmp_checkpoint_dir,
            ignore=shutil.ignore_patterns('dataset', 'async_eval_console'),
        )
        self._save_checkpoint(save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        run_metadata_dict = torch.load(self.ckpt_dir / 'run_metadata.pt')
        self.run_metadata = RunMetadata(**run_metadata_dict)
        self.start_epoch = self.run_metadata.epoch + 1
        self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)
        ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        if self.agent.tokenizer is not None and self.agent.tokenizer.is_trainable:
            self.optimizer_tokenizer.load_state_dict(ckpt_opt['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        self.optimizer_actor_critic.load_state_dict(ckpt_opt['optimizer_actor_critic'])
        self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
        if self.cfg.evaluation.should and not self.is_async_eval_enabled:
            self.test_dataset.num_seen_episodes = torch.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
        logger.info(f'Successfully loaded model, optimizer and {len(self.train_dataset)} episodes from {self.ckpt_dir.absolute()}.')

    def load_best(self):
        assert self.ckpt_dir.is_dir()
        self.agent.load(self.ckpt_dir / 'best.pt', device=self.device)

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for k in batch.keys():
            if isinstance(batch[k], dict):
                batch[k] = self._to_device(batch[k])
            else:
                batch[k] = batch[k].to(self.device)
        return batch

    def finish(self) -> None:
        if self.async_evaluator is not None:
            self.async_evaluator.close()
        close_run_envs(self.train_env, self.test_env)
        wandb.finish()


@hydra.main(config_path="../config", config_name="base", version_base='1.1')
def main(cfg: DictConfig):
    if hasattr(cfg.env, 'mp_spawn_method'):
        multiprocessing.set_start_method(cfg.env.mp_spawn_method)
    train_env, test_env = create_run_envs(cfg)
    try:
        trainer = Trainer(cfg, train_env=train_env, test_env=test_env)
    except Exception:
        close_run_envs(train_env, test_env)
        raise
    trainer.run()


if __name__ == '__main__':
    main()
