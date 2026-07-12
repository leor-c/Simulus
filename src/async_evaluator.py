import multiprocessing
import os
import queue
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import torch.nn as nn
from loguru import logger

from collector import Collector
from dataset import get_dataloader, EpisodeDirManager
from make_reconstructions import make_reconstructions_from_batch
from utils import ObsModality, set_seed


VALID_CONSOLE_MODES = {'hidden', 'file', 'forward'}


@dataclass
class EvalJob:
    epoch: int
    state_dict_path: str
    final: bool = False


@dataclass
class EvalResult:
    epoch: int
    metrics: List[Dict[str, Any]]
    score: Optional[float]
    best_eval_score: Optional[float]
    epoch_of_best_score: Optional[int]
    final: bool = False
    error: Optional[str] = None


@dataclass
class ConsoleEvent:
    stream: str
    text: str


def _async_eval_log_path(ckpt_dir: Path, stream: str) -> Path:
    log_dir = ckpt_dir / 'async_eval_logs'
    log_dir.mkdir(exist_ok=True, parents=True)
    return log_dir / f'{stream}.log'


def _async_eval_fifo_path(ckpt_dir: Path, stream: str) -> Path:
    fifo_dir = ckpt_dir / 'async_eval_console'
    fifo_dir.mkdir(exist_ok=True, parents=True)
    return fifo_dir / f'{stream}.pipe'


def _console_mode(cfg: DictConfig, stream: str) -> str:
    mode = getattr(cfg.evaluation.async_eval.console, stream)
    assert mode in VALID_CONSOLE_MODES, f"Invalid async_eval.console.{stream}: {mode!r}"
    return mode


def _redirect_fd(fd: int, path: Path) -> None:
    redirected_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(redirected_fd, fd)
    os.close(redirected_fd)


def _redirect_to_fifo(fd: int, fifo_path: Optional[str]) -> None:
    assert fifo_path is not None
    fifo_fd = os.open(fifo_path, os.O_WRONLY)
    os.dup2(fifo_fd, fd)
    os.close(fifo_fd)


def redirect_worker_console(cfg: DictConfig, ckpt_dir: Path, stdout_fifo_path: Optional[str], stderr_fifo_path: Optional[str]) -> None:
    """Redirect worker process stdio before native libraries/envs initialize."""
    stdout_mode = _console_mode(cfg, 'stdout')
    stderr_mode = _console_mode(cfg, 'stderr')

    if stdout_mode == 'forward':
        _redirect_to_fifo(1, stdout_fifo_path)
    else:
        stdout_path = Path(os.devnull) if stdout_mode == 'hidden' else _async_eval_log_path(ckpt_dir, 'stdout')
        _redirect_fd(1, stdout_path)

    if stderr_mode == 'forward':
        _redirect_to_fifo(2, stderr_fifo_path)
    else:
        stderr_path = Path(os.devnull) if stderr_mode == 'hidden' else _async_eval_log_path(ckpt_dir, 'stderr')
        _redirect_fd(2, stderr_path)


class EvaluationRunner:
    def __init__(
            self,
            cfg: DictConfig,
            device: torch.device,
            ckpt_dir: Path,
            media_dir: Path,
            best_eval_score: Optional[float],
            epoch_of_best_score: Optional[int],
    ) -> None:
        from main import build_agent, create_env

        self.cfg = cfg
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.media_dir = media_dir
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'
        self.best_eval_score = best_eval_score
        self.epoch_of_best_score = epoch_of_best_score

        torch.set_float32_matmul_precision(cfg.common.float32_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = cfg.common.float32_matmul_precision != 'highest'
        torch.backends.cudnn.allow_tf32 = cfg.common.float32_matmul_precision != 'highest'
        if cfg.common.seed is not None:
            set_seed(cfg.common.seed + 10_000)

        self.test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
        self.test_dataset = instantiate(cfg.datasets.test)
        disable_saving = cfg.common.metrics_only_mode
        episode_manager_test = EpisodeDirManager(
            self.episode_dir / 'test',
            max_num_episodes=cfg.collection.test.num_episodes_to_save,
            disable_saving=disable_saving,
        )
        self.test_collector = Collector(self.test_env, self.test_dataset, episode_manager_test)
        self.agent = build_agent(env=self.test_env, cfg=self.cfg, device=self.device)
        self.agent.eval()

    def close(self) -> None:
        self.test_env.close()

    def run_job(self, job: EvalJob) -> EvalResult:
        state_dict = torch.load(job.state_dict_path, map_location='cpu', weights_only=True)
        self.agent.load_state_dict(state_dict)
        self.agent.to(self.device)
        self.agent.eval()

        self.test_dataset.clear()
        self.test_collector.reset()

        collection_kwargs = self._collection_kwargs(final=job.final)
        test_collect_log = self.test_collector.collect(self.agent, job.epoch, **collection_kwargs)
        metrics = test_collect_log if job.final else test_collect_log + self.eval_agent(job.epoch)

        score_key = f'{self.test_dataset.name}/return'
        score = test_collect_log[-1].get(score_key)
        if score is not None and (self.best_eval_score is None or score >= self.best_eval_score):
            self.best_eval_score = score
            self.epoch_of_best_score = job.epoch
            torch.save(state_dict, self.ckpt_dir / 'best.pt')
            torch.save(
                {
                    'best_eval_score': self.best_eval_score,
                    'epoch_of_best_score': self.epoch_of_best_score,
                },
                self.ckpt_dir / 'best_eval_metadata.pt',
            )

        return EvalResult(
            epoch=job.epoch,
            metrics=metrics,
            score=score,
            best_eval_score=self.best_eval_score,
            epoch_of_best_score=self.epoch_of_best_score,
            final=job.final,
        )

    def _collection_kwargs(self, final: bool) -> Dict[str, Any]:
        collection_kwargs = {**self.cfg.collection.test.config}
        if 'num_episodes' in collection_kwargs:
            if final:
                collection_kwargs['num_episodes'] = collection_kwargs['num_episodes_end']
            kw_to_del = 'num_episodes_end'
        else:
            assert 'num_steps' in collection_kwargs
            if final:
                collection_kwargs['num_steps'] = collection_kwargs['num_steps_end']
            kw_to_del = 'num_steps_end'
        del collection_kwargs[kw_to_del]
        return collection_kwargs

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> List[Dict[str, float]]:
        self.agent.eval()

        metrics_tokenizer, metrics_world_model = {}, {}

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model

        if epoch > cfg_tokenizer.start_after_epochs and self.agent.tokenizer is not None and self.agent.tokenizer.is_trainable:
            metrics_tokenizer = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples, sequence_length=1, context_length=0)

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.eval_component(
                self.agent.world_model,
                cfg_world_model.batch_num_samples,
                sequence_length=self.cfg.common.sequence_length,
                context_length=self.cfg.world_model.context_length,
                tokenizer=self.agent.tokenizer,
            )

        if cfg_tokenizer.save_reconstructions and not self.cfg.common.metrics_only_mode and self.agent.tokenizer is not None and ObsModality.image in self.agent.tokenizer.modalities:
            dataloader = get_dataloader(
                self.test_dataset,
                1,
                self.cfg.common.sequence_length,
                batch_size=3,
                shuffle=True,
                padding_strategy='right',
                obs_modalities=self.agent.tokenizer.modalities,
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
            obs_modalities=self.agent.tokenizer.modalities,
        )
        num_batches = int(np.ceil(len(dataloader) / sequence_length)) if len(dataloader) > sequence_length else len(dataloader)
        data_iter = iter(dataloader)
        for _ in range(num_batches):
            batch = next(data_iter)
            assert (batch['mask_padding'].sum(dim=1) > context_length).all()
            batch = self._to_device(batch)

            losses, _ = component.compute_loss(batch, **kwargs_loss)
            loss_total_epoch += losses.loss_total.item()

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

            steps += 1

        if steps == 0:
            return {}
        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        return {f'{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for k in batch.keys():
            if isinstance(batch[k], dict):
                batch[k] = self._to_device(batch[k])
            else:
                batch[k] = batch[k].to(self.device)
        return batch


def evaluation_worker_loop(
        cfg: DictConfig,
        device_str: str,
        ckpt_dir: str,
        media_dir: str,
        initial_best_eval_score: Optional[float],
        initial_epoch_of_best_score: Optional[int],
        job_queue,
        result_queue,
        stdout_fifo_path: Optional[str],
        stderr_fifo_path: Optional[str],
) -> None:
    runner = None
    ckpt_path = Path(ckpt_dir)
    try:
        redirect_worker_console(cfg, ckpt_path, stdout_fifo_path, stderr_fifo_path)
        runner = EvaluationRunner(
            cfg=cfg,
            device=torch.device(device_str),
            ckpt_dir=ckpt_path,
            media_dir=Path(media_dir),
            best_eval_score=initial_best_eval_score,
            epoch_of_best_score=initial_epoch_of_best_score,
        )
        while True:
            job = job_queue.get()
            if job is None:
                return
            try:
                result = runner.run_job(job)
                if not cfg.evaluation.async_eval.keep_snapshots:
                    Path(job.state_dict_path).unlink(missing_ok=True)
                result_queue.put(result)
            except Exception:
                result_queue.put(EvalResult(
                    epoch=job.epoch,
                    metrics=[],
                    score=None,
                    best_eval_score=runner.best_eval_score,
                    epoch_of_best_score=runner.epoch_of_best_score,
                    final=job.final,
                    error=traceback.format_exc(),
                ))
    except Exception:
        result_queue.put(EvalResult(
            epoch=-1,
            metrics=[],
            score=None,
            best_eval_score=initial_best_eval_score,
            epoch_of_best_score=initial_epoch_of_best_score,
            error=traceback.format_exc(),
        ))
    finally:
        if runner is not None:
            runner.close()


class AsyncEvaluator:
    def __init__(
            self,
            cfg: DictConfig,
            ckpt_dir: Path,
            media_dir: Path,
            best_eval_score: Optional[float],
            epoch_of_best_score: Optional[int],
            save_snapshot: Callable[[int], Path],
    ) -> None:
        self.cfg = cfg
        self.save_snapshot = save_snapshot
        self.pending_jobs = 0
        self._closed = False
        self._console_queue: queue.Queue[ConsoleEvent] = queue.Queue()
        self._reader_threads: List[threading.Thread] = []
        async_cfg = cfg.evaluation.async_eval
        stdout_fifo_path = self._start_console_forwarder(ckpt_dir, 'stdout')
        stderr_fifo_path = self._start_console_forwarder(ckpt_dir, 'stderr')
        ctx = multiprocessing.get_context(async_cfg.start_method)
        self.job_queue = ctx.Queue(maxsize=async_cfg.max_pending_jobs)
        self.result_queue = ctx.Queue()
        self.process = ctx.Process(
            target=evaluation_worker_loop,
            args=(
                cfg,
                async_cfg.device,
                str(ckpt_dir),
                str(media_dir),
                best_eval_score,
                epoch_of_best_score,
                self.job_queue,
                self.result_queue,
                stdout_fifo_path,
                stderr_fifo_path,
            ),
        )
        self.process.start()

    def _start_console_forwarder(self, ckpt_dir: Path, stream: str) -> Optional[str]:
        if _console_mode(self.cfg, stream) != 'forward':
            return None
        fifo_path = _async_eval_fifo_path(ckpt_dir, stream)
        fifo_path.unlink(missing_ok=True)
        os.mkfifo(fifo_path, 0o600)
        read_fd = os.open(fifo_path, os.O_RDWR | os.O_NONBLOCK)
        thread = threading.Thread(
            target=self._read_console_fifo,
            args=(stream, read_fd),
            daemon=True,
        )
        thread.start()
        self._reader_threads.append(thread)
        return str(fifo_path)

    def _read_console_fifo(self, stream: str, read_fd: int) -> None:
        try:
            while not self._closed:
                try:
                    data = os.read(read_fd, 4096)
                except BlockingIOError:
                    time.sleep(0.1)
                    continue
                if data:
                    text = data.decode(errors='replace')
                    self._console_queue.put(ConsoleEvent(stream=stream, text=text))
                else:
                    time.sleep(0.1)
        finally:
            os.close(read_fd)

    def can_dispatch(self) -> bool:
        return self.pending_jobs < self.cfg.evaluation.async_eval.max_pending_jobs

    def dispatch(self, epoch: int, final: bool = False) -> bool:
        if not self.can_dispatch():
            logger.warning(f"Skipping async evaluation for epoch {epoch}; evaluator already has {self.pending_jobs} pending job(s).")
            return False
        state_dict_path = self.save_snapshot(epoch)
        self.job_queue.put(EvalJob(epoch=epoch, state_dict_path=str(state_dict_path), final=final))
        self.pending_jobs += 1
        return True

    def poll(self) -> List[EvalResult]:
        results = []
        while True:
            try:
                result = self.result_queue.get_nowait()
            except Empty:
                break
            if result.epoch != -1:
                self.pending_jobs = max(0, self.pending_jobs - 1)
            results.append(result)
        return results

    def poll_console(self) -> List[ConsoleEvent]:
        events = []
        while True:
            try:
                events.append(self._console_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def wait_for_pending(self) -> List[EvalResult]:
        results = self.poll()
        while self.pending_jobs > 0:
            result = self.result_queue.get()
            self.pending_jobs = max(0, self.pending_jobs - 1)
            results.append(result)
        return results

    def close(self) -> None:
        self._closed = True
        if self.process.is_alive():
            self.job_queue.put(None)
            self.process.join()
