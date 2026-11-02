"""Deploy the web app to a Hugging Face Space (needs `hf auth login` first).

Uploads src/, the trained models (models/distilroberta, models/fast_detect.json), the result tables shown
in the app and the files in deploy/hf_space/. Run from the repository root:

    python deploy/deploy_space.py [--space USER/AI-Text-Flagger]
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ["test_sets.md", "clean_metrics.md", "robustness_table.md", "robustness_chart.png"]


def stage(out: Path) -> None:
    shutil.copytree(ROOT / "deploy" / "hf_space", out, dirs_exist_ok=True)
    shutil.copytree(ROOT / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "models" / "distilroberta", out / "models" / "distilroberta")
    shutil.copy2(ROOT / "models" / "fast_detect.json", out / "models")
    (out / "results").mkdir()
    for name in RESULTS:
        if (ROOT / "results" / name).is_file():
            shutil.copy2(ROOT / "results" / name, out / "results")


def main() -> None:
    api = HfApi()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default=None, help="Space id (default: <your user>/AI-Text-Flagger)")
    args = parser.parse_args()
    space = args.space or f"{api.whoami()['name']}/AI-Text-Flagger"
    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space",
                          commit_message="Deploy AI Text Flagger")
    print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
