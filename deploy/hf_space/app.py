"""Hugging Face Spaces entry point: the same DetectorApp as ``python -m src.app``, served on 0.0.0.0 (CPU)."""

from pathlib import Path

import gradio as gr

from src.app import DetectorApp

ROOT = Path(__file__).parent

app = DetectorApp(models_dir=ROOT / "models", results_dir=ROOT / "results", data_dir=ROOT / "data", device="cpu")
app.build().launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
