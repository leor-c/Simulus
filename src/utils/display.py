from collections import defaultdict
from typing import Optional

import numpy as np
import sys
from tqdm import tqdm
from rich import box
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

_console = Console()


def make_progress_updater(desc: str, total: int, display: Optional['TrainingDisplay'] = None):
    """Return an (n: int) -> None callable that advances a progress bar by n.

    Uses the shared TrainingDisplay when provided, otherwise falls back to tqdm.
    """
    if display is not None:
        task = display.add_task(desc, total=total)
        return lambda n: display.update_task(task, advance=n)
    pbar = tqdm(total=total, desc=desc, file=sys.stdout)
    return pbar.update


def _is_scalar(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating))


def _strip_prefix(k: str, prefix: str) -> str:
    return k[len(prefix) + 1:] if prefix and k.startswith(prefix + '/') else k


def _metrics_table(metrics: dict) -> Optional[Columns]:
    """Build a side-by-side Columns of tables, one per component/phase prefix."""
    groups: dict[str, dict] = defaultdict(dict)
    for k, v in metrics.items():
        if not _is_scalar(v):
            continue
        parts = k.split('/')
        prefix = '/'.join(parts[:2]) if len(parts) >= 3 else ''
        name = '/'.join(parts[2:]) if len(parts) >= 3 else k
        groups[prefix][name] = v
    if not groups:
        return None
    tables = []
    for prefix, group in groups.items():
        table = Table(title=prefix or 'metrics', box=box.SIMPLE_HEAD, header_style='bold magenta', show_edge=False)
        table.add_column('metric', style='dim', no_wrap=True)
        table.add_column('value', justify='right')
        for name, v in group.items():
            table.add_row(name, f'{v:.4f}' if isinstance(v, float) else str(v))
        tables.append(table)
    return Columns(tables, equal=False, expand=False)


def _collector_dataset_name(logs: list[dict]) -> str:
    for d in logs:
        for k in d.keys():
            if '/' in k:
                return k.split('/')[0]
    return ''


def _collector_summary_text(logs: list[dict]) -> str:
    """One-line summary from the collector's final summary dict."""
    dataset_name = _collector_dataset_name(logs)
    summary_dict = next((d for d in reversed(logs) if any('#' in k for k in d)), {})
    if not summary_dict:
        return ''
    parts = [
        f'{_strip_prefix(k, dataset_name)}: {v:.3f}' if isinstance(v, float)
        else f'{_strip_prefix(k, dataset_name)}: {v}'
        for k, v in summary_dict.items() if _is_scalar(v)
    ]
    return f'[bold cyan]{dataset_name}[/bold cyan]  {" · ".join(parts)}' if parts else ''


def print_collector_log(logs: list[dict]) -> None:
    """Print collector episode rows as a table + summary (used in per_epoch mode)."""
    if not logs:
        return
    dataset_name = _collector_dataset_name(logs)

    def strip(k: str) -> str:
        return _strip_prefix(k, dataset_name)

    episode_dicts = [d for d in logs if any('episode_num' in k for k in d)]
    summary_dict = next((d for d in reversed(logs) if any('#' in k for k in d)), {})

    if episode_dicts:
        col_keys = [(k, strip(k)) for k, v in episode_dicts[0].items() if _is_scalar(v)]
        if col_keys:
            table = Table(title=f'{dataset_name} episodes', box=box.SIMPLE_HEAD, header_style='bold cyan', show_edge=False)
            for _, col_name in col_keys:
                table.add_column(col_name, justify='right')
            for d in episode_dicts:
                row = []
                for orig_k, _ in col_keys:
                    v = d.get(orig_k)
                    if v is None:
                        row.append('—')
                    elif isinstance(v, float):
                        row.append(f'{v:.3f}')
                    else:
                        row.append(str(v))
                table.add_row(*row)
            _console.print(table)

    if summary_dict:
        parts = [
            f'{strip(k)}: {v:.3f}' if isinstance(v, float) else f'{strip(k)}: {v}'
            for k, v in summary_dict.items() if _is_scalar(v)
        ]
        if parts:
            _console.print(f'  [dim]{" · ".join(parts)}[/dim]\n')


class TrainingDisplay:
    """Live-updating rich display for training progress and metrics.

    mode="per_epoch"  - each epoch's final state is committed to the terminal;
                        next epoch renders below it (scrollable history).
    mode="persistent" - a single Live panel covers the whole run; each epoch
                        overwrites the previous one (clean terminal).
    mode="disabled"   - track tasks internally without rendering.
    """

    def __init__(self, mode: str = 'per_epoch'):
        assert mode in ('per_epoch', 'persistent', 'disabled'), f"Invalid display_mode: {mode!r}"
        self.mode = mode
        self._live: Optional[Live] = None
        self._progress: Optional[Progress] = None
        self._metrics: dict = {}
        self._status: str = ''
        self._collector_summaries: dict[str, str] = {}
        self._info: str = ''
        self._epoch = 0
        self._total_epochs = 0

    # ------------------------------------------------------------------
    # Context manager for the whole training run (needed for persistent)
    # ------------------------------------------------------------------

    def __enter__(self):
        if self.mode == 'persistent':
            self._start_live()
        return self

    def __exit__(self, *_):
        if self._live:
            self._live.stop()
            self._live = None

    # ------------------------------------------------------------------
    # Per-epoch lifecycle
    # ------------------------------------------------------------------

    def start_epoch(self, epoch: int, total_epochs: int) -> None:
        self._epoch = epoch
        self._total_epochs = total_epochs
        self._status = ''
        self._info = ''
        if self.mode == 'disabled':
            self._reset_progress()
            return
        if self.mode == 'per_epoch':
            self._metrics = {}
            self._start_live()
        else:
            # Keep previous epoch's metrics visible until new ones arrive.
            self._reset_progress()
            self._refresh()

    def end_epoch(self, status: str = '') -> None:
        self._status = status
        if self.mode == 'disabled':
            self._progress = None
            return
        if self.mode == 'per_epoch' and self._live:
            self._refresh()  # commit final state including status
            self._live.stop()
            self._live = None
            self._progress = None
        else:
            self._refresh()

    # ------------------------------------------------------------------
    # Progress tasks
    # ------------------------------------------------------------------

    def add_task(self, description: str, total: int) -> TaskID:
        assert self._progress is not None, "Call start_epoch before add_task"
        return self._progress.add_task(description, total=total, metrics='')

    def update_task(self, task_id: TaskID, advance: int = 1, metrics: str = '') -> None:
        assert self._progress is not None
        self._progress.update(task_id, advance=advance, metrics=metrics)
        task = self._progress.tasks[task_id]
        if task.completed >= task.total:
            self._progress.update(task_id, visible=False)
        self._refresh()

    # ------------------------------------------------------------------
    # Metrics table
    # ------------------------------------------------------------------

    def update_metrics(self, new_metrics: dict) -> None:
        self._metrics.update({k: v for k, v in new_metrics.items() if _is_scalar(v)})
        self._refresh()

    def update_info(self, text: str) -> None:
        """Show a persistent info line inside the renderable (no external print)."""
        self._info = text
        self._refresh()

    def show_collector(self, logs: list[dict]) -> None:
        """Show collector results.

        per_epoch: prints the full episode table to the console (before Live starts).
        persistent: stores a one-line summary inside the Live renderable.
        """
        if self.mode == 'per_epoch':
            print_collector_log(logs)
        elif self.mode == 'disabled':
            return
        else:
            dataset_name = _collector_dataset_name(logs)
            self._collector_summaries[dataset_name] = _collector_summary_text(logs)
            self._refresh()

    @property
    def console(self) -> Console:
        """Routes through the live console in persistent mode to avoid committing the live header."""
        if self._live is not None and self.mode == 'persistent':
            return self._live.console
        return _console

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make_progress() -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn('[bold]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn('[cyan]{task.fields[metrics]}'),
            auto_refresh=False,
        )

    def _reset_progress(self) -> None:
        self._progress = self._make_progress()

    def _start_live(self) -> None:
        self._progress = self._make_progress()
        self._live = Live(
            self._render(),
            auto_refresh=False,
            transient=(self.mode == 'persistent'),
        )
        self._live.start()

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())
            self._live.refresh()

    def _render(self) -> Group:
        parts: list[RenderableType] = [Text(f'Epoch {self._epoch} / {self._total_epochs}', style='bold')]
        for summary in self._collector_summaries.values():
            if summary:
                parts.append(Text.from_markup(summary))
        if self._info:
            parts.append(Text(self._info, style='dim'))
        if self._progress:
            parts.append(self._progress)
        columns = _metrics_table(self._metrics)
        if columns:
            parts.append(columns)
        if self._status:
            parts.append(Text(self._status, style='dim'))
        return Group(*parts)
