"""Run the complete offline pipeline in order (report Figure 1 and Section 3.3).

Steps
-----
1. ``data``         - HC3 + MAGE download/clean/split    (``python -m src.data``)
2. ``zero-shot``    - GPT-2 statistics + calibrators     (``python -m src.zero_shot``)
3. ``fast-detect``  - Fast-DetectGPT scores + calibrator (``python -m src.fast_detect``)
4. ``finetune``     - DistilRoBERTa fine-tuning          (``python -m src.finetune``)
5. ``clean``        - HC3 test-set metrics               (``python -m src.evaluate --clean``)
6. ``sets``         - all test sets + before/after (v1)  (``python -m src.evaluate --sets``)
7. ``robust``       - perturbation robustness (AUA)      (``python -m src.evaluate --robust``)
8. ``app``          - build the web UI and time one detection without starting a server

The Modern-AI test set (``data/test_modern.csv``) needs a free Groq key and is built separately with
``python -m src.modern_set --env-file PATH``; the ``sets`` step uses it when it exists.

Every step reads and writes ``<work dir>/data``, ``<work dir>/models`` and ``<work dir>/results``
(GPT-2 cache in ``<work dir>/results/cache``). The work dir is the repository root for the full-size
run and ``smoke/`` for ``--smoke``, so a smoke run never replaces the full-size results. Any other
reduced run (``--limit``, another ``--k`` or ``--epochs``) must name its own ``--work-dir``.

Examples::

    python run_pipeline.py                 # full run (several hours of GPU time on an RTX 2050, see README)
    python run_pipeline.py --smoke         # tiny end-to-end check in smoke/: 64 rows, 1 epoch, k = 2
    python run_pipeline.py --skip-data --skip-zero-shot --skip-finetune --skip-clean   # resume: robust + app
    python run_pipeline.py --limit 500 --k 3 --work-dir runs/small                     # a reduced run
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from src.base import configure_console, free_gpu_memory, refuse_reduced_run
from src.config import (
    FAST_DETECT_JSON,
    FD_BATCH_SIZE,
    FINETUNED_DIR,
    FT_BATCH_SIZE,
    FT_EPOCHS,
    HC3_DEFAULT_PATH,
    K_VARIANTS,
    ROOT_DIR,
    SMOKE_DIR,
    SMOKE_EPOCHS,
    SMOKE_K,
    SMOKE_LIMIT,
    ZERO_SHOT_JSON,
    ZS_BATCH_SIZE,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run_pipeline.py",
        description="Run data -> zero-shot -> fast-detect -> fine-tune -> clean -> test sets -> robustness -> app check.",
    )
    parser.add_argument("--smoke", action="store_true",
                        help=f"tiny end-to-end run in {SMOKE_DIR.name}/: limit {SMOKE_LIMIT} rows, "
                             f"{SMOKE_EPOCHS} epoch, k = {SMOKE_K}")
    parser.add_argument("--limit", type=int, default=None, help="cap every split at N rows (needs --work-dir)")
    parser.add_argument("--k", type=int, default=None,
                        help=f"variants per text/operation/level (default: {K_VARIANTS}; another value needs --work-dir)")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"fine-tuning epochs (default: {FT_EPOCHS}; another value needs --work-dir)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"batch size for all steps (default: {ZS_BATCH_SIZE} GPT-2 / {FT_BATCH_SIZE} DistilRoBERTa)")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="folder that holds data/, models/ and results/ of this run (default: the "
                             f"repository root; {SMOKE_DIR.name}/ with --smoke)")
    parser.add_argument("--hc3-path", type=Path, default=HC3_DEFAULT_PATH, help="HC3 all.jsonl (default: %(default)s)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    for step in ("data", "zero-shot", "fast-detect", "finetune", "clean", "sets", "robust", "app"):
        parser.add_argument(f"--skip-{step}", action="store_true", help=f"skip the {step} step")
    return parser


def settings(args: argparse.Namespace) -> tuple[int | None, int, int]:
    """The run's ``(limit, k, epochs)``: explicit flags, else the smoke or the full-size values."""
    limit = args.limit if args.limit is not None else (SMOKE_LIMIT if args.smoke else None)
    k = args.k if args.k is not None else (SMOKE_K if args.smoke else K_VARIANTS)
    epochs = args.epochs if args.epochs is not None else (SMOKE_EPOCHS if args.smoke else FT_EPOCHS)
    return limit, k, epochs


def reduced_reasons(args: argparse.Namespace) -> list[str]:
    """How this run differs from the full-size configuration (empty for the full-size run)."""
    limit, k, epochs = settings(args)
    reasons = [f"--limit {limit}"] if limit is not None else []
    if k != K_VARIANTS:
        reasons.append(f"k = {k}")
    if epochs != FT_EPOCHS:
        reasons.append(f"{epochs} epoch(s)")
    return reasons


def work_dir(args: argparse.Namespace) -> Path:
    """Folder holding this run's data/, models/ and results/ (see the module docstring)."""
    if args.work_dir is not None:
        return args.work_dir
    return SMOKE_DIR if args.smoke else ROOT_DIR


def plan(args: argparse.Namespace) -> list[tuple[str, Callable[[Sequence[str]], int], list[str]]]:
    """Return the (name, entry point, argv) list of steps to run."""
    from src import app, data, evaluate, fast_detect, finetune, zero_shot

    limit, k, epochs = settings(args)
    root = work_dir(args)
    data_dir, models_dir, results_dir = root / "data", root / "models", root / "results"
    cache_dir = results_dir / "cache"
    common = ["--limit", str(limit)] if limit is not None else []
    device = ["--device", args.device] if args.device else []
    zs_batch = ["--batch-size", str(args.batch_size or ZS_BATCH_SIZE)]
    ft_batch = ["--batch-size", str(args.batch_size or FT_BATCH_SIZE)]
    evaluation = ["--data-dir", str(data_dir), "--models-dir", str(models_dir),
                  "--results-dir", str(results_dir), "--cache-dir", str(cache_dir)]
    steps = [
        ("data", data.main, ["--hc3-path", str(args.hc3_path), "--data-dir", str(data_dir), *common]),
        ("zero-shot", zero_shot.main, [*common, *zs_batch, "--data-dir", str(data_dir),
                                       "--output", str(models_dir / ZERO_SHOT_JSON.name),
                                       "--cache-dir", str(cache_dir), *device]),
        ("fast-detect", fast_detect.main, [*common, "--batch-size", str(FD_BATCH_SIZE), "--data-dir", str(data_dir),
                                           "--output", str(models_dir / FAST_DETECT_JSON.name),
                                           "--cache-dir", str(cache_dir), *device]),
        ("finetune", finetune.main, [*common, *ft_batch, "--epochs", str(epochs), "--data-dir", str(data_dir),
                                     "--output-dir", str(models_dir / FINETUNED_DIR.name), *device]),
        ("clean", evaluate.main, ["--clean", *common, *zs_batch, *evaluation, *device]),
        ("sets", evaluate.main, ["--sets", *common, *zs_batch, *evaluation, *device]),
        ("robust", evaluate.main, ["--robust", "--k", str(k), *common, *zs_batch, *evaluation, *device]),
        ("app", app.main, ["--dry-run", "--data-dir", str(data_dir), "--models-dir", str(models_dir),
                           "--results-dir", str(results_dir), *device]),
    ]
    return [step for step in steps if not getattr(args, f"skip_{step[0].replace('-', '_')}")]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected steps; stop at the first failure."""
    configure_console()
    args = build_arg_parser().parse_args(argv)
    # A reduced run may only use the repository's data/, models/ and results/ (the full-size run)
    # when --work-dir names them on purpose; --smoke gets its own folder by default.
    if not args.smoke and refuse_reduced_run("pipeline", reduced_reasons(args),
                                             [] if args.work_dir else [("--work-dir", ROOT_DIR)]):
        return 2
    print(f"[pipeline] work dir: {work_dir(args).resolve()}", flush=True)
    timings = []
    for name, entry, step_argv in plan(args):
        print(f"\n{'=' * 78}\n[pipeline] step '{name}': {' '.join(step_argv)}\n{'=' * 78}", flush=True)
        started = time.perf_counter()
        code = entry(step_argv)
        timings.append((name, time.perf_counter() - started))
        free_gpu_memory()
        if code != 0:
            print(f"[pipeline] step '{name}' failed with exit code {code}")
            return code
    print(f"\n[pipeline] finished{' (smoke run)' if args.smoke else ''}:")
    for name, seconds in timings:
        print(f"  {name:<12} {seconds:8.1f} s")
    print(f"  {'total':<12} {sum(s for _, s in timings):8.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
