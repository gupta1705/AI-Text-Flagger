"""The smoke run and other reduced runs must never overwrite the full-size outputs.

Every test points the modules' default output paths at ``tmp_path`` first, so even a failing test
cannot touch the real ``data/``, ``models/`` or ``results/`` folders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import run_pipeline
from src import data as data_module
from src import evaluate as evaluate_module
from src import finetune as finetune_module
from src import zero_shot as zero_shot_module
from src.config import DATA_DIR, K_VARIANTS, MODELS_DIR, RESULTS_DIR, ROOT_DIR, SMOKE_DIR
from src.data import HC3DataModule


def _argv_of(steps) -> dict[str, list[str]]:
    return {name: argv for name, _, argv in steps}


def _value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def _all_paths(steps) -> list[str]:
    return [v for _, _, argv in steps for v in argv if "/" in v or "\\" in v]


def test_smoke_plan_writes_only_into_the_smoke_folder() -> None:
    args = run_pipeline.build_arg_parser().parse_args(["--smoke"])
    steps = _argv_of(run_pipeline.plan(args))
    assert list(steps) == ["data", "zero-shot", "fast-detect", "finetune", "clean", "sets", "robust", "app"]
    assert _value(steps["data"], "--data-dir") == str(SMOKE_DIR / "data")
    assert _value(steps["zero-shot"], "--output") == str(SMOKE_DIR / "models" / "zero_shot.json")
    assert _value(steps["zero-shot"], "--cache-dir") == str(SMOKE_DIR / "results" / "cache")
    assert _value(steps["fast-detect"], "--output") == str(SMOKE_DIR / "models" / "fast_detect.json")
    assert _value(steps["fast-detect"], "--cache-dir") == str(SMOKE_DIR / "results" / "cache")
    assert _value(steps["finetune"], "--output-dir") == str(SMOKE_DIR / "models" / "distilroberta")
    for name in ("clean", "sets", "robust"):
        assert _value(steps[name], "--results-dir") == str(SMOKE_DIR / "results")
        assert _value(steps[name], "--models-dir") == str(SMOKE_DIR / "models")
    assert _value(steps["app"], "--models-dir") == str(SMOKE_DIR / "models")
    assert _value(steps["robust"], "--k") == "2" and _value(steps["finetune"], "--epochs") == "1"
    # Apart from the HC3 input file, no path outside smoke/ appears in any step.
    outputs = [p for p in _all_paths(run_pipeline.plan(args)) if p != str(args.hc3_path)]
    assert outputs and all(p.startswith(str(SMOKE_DIR)) for p in outputs)


def test_full_plan_uses_the_repository_folders() -> None:
    args = run_pipeline.build_arg_parser().parse_args([])
    steps = _argv_of(run_pipeline.plan(args))
    assert _value(steps["data"], "--data-dir") == str(DATA_DIR)
    assert _value(steps["clean"], "--results-dir") == str(RESULTS_DIR)
    assert _value(steps["robust"], "--models-dir") == str(MODELS_DIR)
    assert _value(steps["robust"], "--k") == str(K_VARIANTS)
    assert all("--limit" not in argv for argv in steps.values())
    assert run_pipeline.reduced_reasons(args) == []


def test_resume_plan_runs_only_robust_and_app() -> None:
    args = run_pipeline.build_arg_parser().parse_args(
        ["--skip-data", "--skip-zero-shot", "--skip-fast-detect", "--skip-finetune", "--skip-clean", "--skip-sets"])
    assert [name for name, _, _ in run_pipeline.plan(args)] == ["robust", "app"]
    assert run_pipeline.work_dir(args) == ROOT_DIR


@pytest.mark.parametrize("argv", [["--limit", "100"], ["--k", "5"], ["--epochs", "1"]])
def test_reduced_pipeline_run_needs_its_own_work_dir(argv, monkeypatch, capsys) -> None:
    def must_not_run(args):
        raise AssertionError("a refused run must not start any step")

    monkeypatch.setattr(run_pipeline, "plan", must_not_run)
    assert run_pipeline.main(argv) == 2
    assert "--work-dir" in capsys.readouterr().err


def test_reduced_pipeline_run_with_a_work_dir_is_allowed(tmp_path, monkeypatch) -> None:
    planned = []
    monkeypatch.setattr(run_pipeline, "plan", lambda args: planned.append(run_pipeline.work_dir(args)) or [])
    assert run_pipeline.main(["--limit", "100", "--work-dir", str(tmp_path / "small")]) == 0
    assert planned == [tmp_path / "small"]


def test_data_limit_needs_its_own_data_dir(hc3_file: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    default = tmp_path / "default_data"
    monkeypatch.setattr(data_module, "DATA_DIR", default)
    assert data_module.main(["--hc3-path", str(hc3_file), "--limit", "6", "--no-mage"]) == 2
    assert "--data-dir" in capsys.readouterr().err and not default.exists()
    own = tmp_path / "own_data"
    assert data_module.main(["--hc3-path", str(hc3_file), "--limit", "6", "--no-mage", "--data-dir", str(own)]) == 0
    assert len(HC3DataModule.load_split("test", own)) == 6 and not default.exists()


@pytest.fixture()
def small_splits(hc3_file: Path, tmp_path: Path) -> Path:
    """Tiny train/val/test CSVs (fewer rows than the full-size splits)."""
    data_dir = tmp_path / "small_data"
    module = HC3DataModule(hc3_path=hc3_file, data_dir=data_dir, cached_path=None, limit=6)
    module.save(module.make_splits(module.clean(module.load())))
    return data_dir


def test_zero_shot_on_small_splits_needs_its_own_output(small_splits, tmp_path, monkeypatch, capsys) -> None:
    default = tmp_path / "models" / "zero_shot.json"
    monkeypatch.setattr(zero_shot_module, "ZERO_SHOT_JSON", default)
    monkeypatch.setattr(zero_shot_module, "ZeroShotDetector", None)        # GPT-2 must not be loaded
    argv = ["--data-dir", str(small_splits), "--cache-dir", str(tmp_path / "cache")]
    assert zero_shot_module.main(argv) == 2
    assert "--output" in capsys.readouterr().err and not default.exists()


def test_finetune_on_small_splits_needs_its_own_output_dir(small_splits, tmp_path, monkeypatch, capsys) -> None:
    default = tmp_path / "models" / "distilroberta"
    monkeypatch.setattr(finetune_module, "FINETUNED_DIR", default)
    monkeypatch.setattr(finetune_module, "FineTunedDetector", None)        # nothing is trained
    assert finetune_module.main(["--data-dir", str(small_splits)]) == 2
    assert "--output-dir" in capsys.readouterr().err and not default.exists()


def test_reduced_evaluation_needs_its_own_results_dir(small_splits, tmp_path, monkeypatch, capsys) -> None:
    default = tmp_path / "default_results"
    monkeypatch.setattr(evaluate_module, "RESULTS_DIR", default)
    argv = ["--robust", "--k", "2", "--detectors", "loglik,distilroberta", "--data-dir", str(small_splits),
            "--cache-dir", str(tmp_path / "cache")]
    assert evaluate_module.main(argv) == 2
    err = capsys.readouterr().err
    assert "k = 2" in err and "2 of 6 detectors" in err and "--results-dir" in err
    assert not default.exists()
    # With its own folder the run goes ahead; here it stops cleanly because no models exist.
    own = ["--results-dir", str(tmp_path / "own"), "--models-dir", str(tmp_path / "no_models")]
    assert evaluate_module.main(argv + own) == 1
    assert "not found" in capsys.readouterr().err and not default.exists()


def test_missing_inputs_give_a_message_not_a_traceback(tmp_path, capsys) -> None:
    empty = str(tmp_path / "empty")
    assert zero_shot_module.main(["--data-dir", empty]) == 1
    assert finetune_module.main(["--data-dir", empty]) == 1
    assert evaluate_module.main(["--clean", "--data-dir", empty]) == 1
    assert capsys.readouterr().err.count("python -m src.data") == 3
