"""DetectorApp: local Gradio web application (report Section 3.5, Algorithm 5, Listing 4, Figure 5).

The app combines one detector of each kind:

* the fine-tuned DistilRoBERTa classifier (P(AI) averaged over 256-token chunks), and
* the Fast-DetectGPT zero-shot detector (v2; Qwen2.5-0.5B scoring model, src/fast_detect.py).

Decision rule (:func:`decide`, a pure function): if both probabilities are on the same side of 0.5
the verdict is "Likely AI-generated" / "Likely human-written" with the mean probability, otherwise
"Uncertain – review manually". Texts under 20 words get the note "Short text: result is less
reliable." A text with fewer than two tokens has no zero-shot score (NaN); the detectors
then cannot agree, so the verdict is "Uncertain" with a note saying why. Input is normalised
exactly like the training data before scoring.

Privacy: the pasted text is only held in memory for scoring; it is never written to disk, never
logged and never sent anywhere (Gradio analytics are disabled and the server binds 127.0.0.1).

CLI: ``python -m src.app [--port 7860] [--share false] [--dry-run] [--models-dir D] [--results-dir D] [--data-dir D]``
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import argparse
import html
import json
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
import pandas as pd

from src.base import configure_console
from src.config import (
    APP_HOST,
    APP_PORT,
    APP_TITLE,
    CLEAN_METRICS_MD,
    DATA_DIR,
    DETECTOR_DISPLAY,
    FAST_DETECT_JSON,
    FINETUNED_DIR,
    LABEL_AI,
    LABEL_HUMAN,
    MODELS_DIR,
    RESULTS_DIR,
    ROBUSTNESS_CHART_PNG,
    ROBUSTNESS_MD,
    SHORT_TEXT_WORDS,
    TEST_SETS_MD,
    THRESHOLD,
    VERDICT_AI,
    VERDICT_HUMAN,
    VERDICT_UNCERTAIN,
)
from src.text_utils import normalize_text, word_count

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.fast_detect import FastDetectGPTDetector
    from src.finetune import FineTunedDetector

SHORT_TEXT_NOTE = "Short text: result is less reliable."
NO_ZERO_SHOT_NOTE = "Too short for the zero-shot detector (it needs at least two tokens), so the detectors cannot be compared."
LIMITATIONS_NOTE = (
    "This is a probability, not proof. Accuracy drops for very short texts, text with many typos, "
    "text heavily edited by a person, and topics or AI models not seen in training. "
    "The pasted text is not stored."
)
BADGE_COLORS = {VERDICT_AI: "#c53030", VERDICT_HUMAN: "#2f855a", VERDICT_UNCERTAIN: "#b7791f"}


@dataclass(frozen=True)
class Verdict:
    """Result shown to the user (report Listing 4)."""

    label: str
    p_finetuned: float
    p_zeroshot: float
    agreement: bool
    note: str

    @property
    def has_zeroshot(self) -> bool:
        """False when the text was too short for the zero-shot detector (``p_zeroshot`` is NaN)."""
        return not math.isnan(self.p_zeroshot)

    @property
    def mean_probability(self) -> float:
        """Mean P(AI) of the two detectors (NaN without a zero-shot score)."""
        return (self.p_finetuned + self.p_zeroshot) / 2


def decide(p_finetuned: float, p_zeroshot: float, n_words: int, threshold: float = THRESHOLD) -> Verdict:
    """Combine both detectors (Algorithm 5); disagreement is reported as 'Uncertain'.

    ``p_zeroshot`` may be NaN: the zero-shot detector had no evidence (fewer than two
    tokens). The detectors then cannot agree, so the verdict is 'Uncertain' with a note.
    """
    if not 0.0 <= p_finetuned <= 1.0:            # also rejects NaN
        raise ValueError(f"p_finetuned must be a probability in [0, 1], got {p_finetuned}")
    if not (math.isnan(p_zeroshot) or 0.0 <= p_zeroshot <= 1.0):
        raise ValueError(f"p_zeroshot must be a probability in [0, 1] or NaN, got {p_zeroshot}")
    notes = [SHORT_TEXT_NOTE] if n_words < SHORT_TEXT_WORDS else []
    ft_ai = p_finetuned >= threshold
    if math.isnan(p_zeroshot):
        agreement = False
        notes.append(NO_ZERO_SHOT_NOTE)
    else:
        agreement = ft_ai == (p_zeroshot >= threshold)
    if not agreement:
        label = VERDICT_UNCERTAIN
    else:
        label = VERDICT_AI if ft_ai else VERDICT_HUMAN
    return Verdict(label, float(p_finetuned), float(p_zeroshot), agreement, " ".join(notes))


# ---------------------------------------------------------------------- HTML rendering
def verdict_badge_html(verdict: Verdict) -> str:
    """Coloured verdict badge (red AI / green human / amber uncertain) with the mean probability."""
    color = BADGE_COLORS[verdict.label]
    if verdict.has_zeroshot:
        detail = f"Mean probability of AI generation: <b>{verdict.mean_probability:.0%}</b>"
    else:
        detail = f"No zero-shot score; fine-tuned P(AI): <b>{verdict.p_finetuned:.0%}</b>"
    return (
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin:4px 0">'
        f'<span style="background:{color};color:#fff;font-weight:700;font-size:1.15rem;'
        f'padding:8px 18px;border-radius:999px">{html.escape(verdict.label)}</span>'
        f'<span>{detail}</span></div>'
    )


def probability_bars_html(verdict: Verdict, zero_shot_name: str) -> str:
    """One horizontal bar per detector with the 0.5 decision line marked."""
    rows = []
    for name, p in (("Fine-tuned detector (DistilRoBERTa)", verdict.p_finetuned),
                    (f"Zero-shot detector ({zero_shot_name})", verdict.p_zeroshot)):
        missing = math.isnan(p)                # zero-shot detector had no evidence
        fill = "#c53030" if not missing and p >= THRESHOLD else "#2f855a"
        value = "n/a (text too short)" if missing else f"{p:.0%}"
        width = 0.0 if missing else p * 100
        rows.append(
            f'<div style="margin:10px 0">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.95rem">'
            f'<span>{html.escape(name)}</span><b>P(AI) = {value}</b></div>'
            f'<div style="position:relative;height:14px;border-radius:7px;background:rgba(128,128,128,0.25);'
            f'overflow:hidden;margin-top:4px">'
            f'<div style="width:{width:.1f}%;height:100%;background:{fill}"></div>'
            f'<div style="position:absolute;left:50%;top:0;bottom:0;width:2px;background:rgba(0,0,0,0.55)"></div>'
            f'</div></div>'
        )
    return "".join(rows)


def agreement_text(verdict: Verdict) -> str:
    """Agreement line shown under the probability bars."""
    if not verdict.has_zeroshot:
        return ("**Agreement:** cannot be checked - the text is too short for the zero-shot detector, "
                "so the result is *Uncertain*.")
    if verdict.agreement:
        return "**Agreement:** both detectors are on the same side of 0.5."
    return "**Agreement:** the detectors disagree, so the result is *Uncertain* and needs manual review."


def word_count_text(text: str) -> str:
    """Live word count shown under the input box (counted after normalisation)."""
    n = word_count(normalize_text(text))
    hint = f" - under {SHORT_TEXT_WORDS} words, the result will be less reliable" if 0 < n < SHORT_TEXT_WORDS else ""
    return f"{n} word{'s' if n != 1 else ''}{hint}"


ABOUT_MARKDOWN = f"""
## How the two detectors work (v2)

**Fine-tuned detector (DistilRoBERTa).** A DistilRoBERTa transformer (82 M parameters) was trained for
up to three epochs on 13,500 human and AI texts: 4,500 from HC3 (ChatGPT answers to questions) and
9,000 from MAGE (stories, opinion essays, reviews, news, scientific writing and more, written by 27
different AI models). Long texts are split into 256-token chunks and the AI probabilities are averaged.

**Zero-shot detector (Fast-DetectGPT).** The text is read by a small modern language model
(Qwen2.5-0.5B, first 256 tokens). For every word position the detector compares how likely the actual
word is with how likely the model's own typical choices would be. AI text stays close to what a model
would write; human text does not. It needs no training on detection data, so it does not learn the
quirks of one dataset; a logistic calibration turns the score into a probability.

**Verdict.** If both probabilities are on the same side of 0.5, the matching verdict is shown with
the mean probability; otherwise the result is *{VERDICT_UNCERTAIN}*. A text of fewer than two tokens
has no zero-shot score, so it is always *{VERDICT_UNCERTAIN}*.

## Data

Training: HC3 English (Guo et al., 2023; CC-BY-SA-4.0) and MAGE (Li et al., 2024; Apache-2.0).
Testing: the HC3 test split, a MAGE test sample, MAGE's out-of-distribution set (CNN news, IMDB
reviews, PubMed abstracts and dialogue summaries by GPT-4) and a *Modern AI* set of same-topic texts
written by 2025-26 models (Qwen3.8-27B, gpt-oss-20b) paired with human texts. All texts are
normalised the same way so that formatting quirks of the datasets (HC3's pre-tokenised human answers,
MAGE's forum footers and cut-off machine continuations) do not give the class away.

## Limitations

- {LIMITATIONS_NOTE}
- New AI models and writing styles keep appearing; the *Model report* tab shows how accuracy differs
  between familiar and unfamiliar kinds of text.
- Text written by non-native English speakers may be flagged more often (Liang et al., 2023).
- Use the result as supporting evidence for a conversation, never as the sole basis for a decision.

## Project

CSET312 AI/ML mini-project, School of Computer Science Engineering and Technology, Bennett University
(Dec 2026). Team: Vartika Manish (S24CSEU0169) and Vedansh Gupta (S24CSEU0173). Mentor: Himanshu Sharma.

Base paper: X. Liu, Y. Li and K. Li, "Enhancing the Robustness of AI-Generated Text Detectors:
A Survey", *Mathematics* 13(13), 2145, 2025. doi:10.3390/math13132145
"""


class DetectorApp:
    """Loads both detectors lazily (once) and serves them through a Gradio Blocks UI.

    Args:
        models_dir: folder holding ``fast_detect.json`` and ``distilroberta/``.
        results_dir: folder holding the report tables and chart.
        data_dir: folder holding ``test.csv`` (used for the two example texts).
        device: ``"cuda"``/``"cpu"`` (default: CUDA if available, otherwise CPU).
    """

    def __init__(
        self,
        models_dir: str | Path = MODELS_DIR,
        results_dir: str | Path = RESULTS_DIR,
        data_dir: str | Path = DATA_DIR,
        device: str | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        self.device = device
        self._lock = threading.Lock()
        self._finetuned: FineTunedDetector | None = None
        self._zeroshot: FastDetectGPTDetector | None = None

    # ------------------------------------------------------------------ models
    @property
    def zero_shot_path(self) -> Path:
        return self.models_dir / FAST_DETECT_JSON.name

    @property
    def finetuned_path(self) -> Path:
        return self.models_dir / FINETUNED_DIR.name

    def models_available(self) -> bool:
        """True when both trained detectors exist on disk."""
        return self.zero_shot_path.is_file() and (self.finetuned_path / "config.json").is_file()

    def _load_models(self) -> tuple["FineTunedDetector", "FastDetectGPTDetector"]:
        with self._lock:
            if self._finetuned is None or self._zeroshot is None:
                from src.fast_detect import FastDetectGPTDetector
                from src.finetune import FineTunedDetector

                finetuned = FineTunedDetector.load(self.finetuned_path, device=self.device)
                zeroshot = FastDetectGPTDetector.load(self.zero_shot_path, device=self.device)
                finetuned.load_model()
                zeroshot.load_model()
                self._finetuned, self._zeroshot = finetuned, zeroshot
        return self._finetuned, self._zeroshot

    def zero_shot_name(self) -> str:
        """Display name of the zero-shot detector, e.g. ``Fast-DetectGPT, Qwen2.5-0.5B`` (read from the JSON)."""
        try:
            with open(self.zero_shot_path, encoding="utf-8") as handle:
                model = json.load(handle)["model_name"]
        except (OSError, KeyError, json.JSONDecodeError):
            return "Fast-DetectGPT"
        return f"Fast-DetectGPT, {model.split('/')[-1]}"

    def detect(self, text: str) -> Verdict:
        """Normalise ``text``, score it with both detectors and apply :func:`decide`."""
        clean = normalize_text(text)
        if not clean:
            raise ValueError("Please paste some text to check.")
        finetuned, zeroshot = self._load_models()
        p_ft = float(finetuned.predict_proba([clean])[0])
        p_zs = float(zeroshot.predict_proba([clean])[0])
        return decide(p_ft, p_zs, word_count(clean))

    # ------------------------------------------------------------------ content
    def examples(self) -> list[tuple[str, str]]:
        """One human and one ChatGPT answer from ``test.csv`` (empty list if it is missing)."""
        path = self.data_dir / "test.csv"
        if not path.is_file():
            return []
        frame = pd.read_csv(path, encoding="utf-8", dtype={"text": str}, keep_default_na=False)
        picked = []
        for label, caption in ((LABEL_HUMAN, "Human answer (HC3 test set)"),
                               (LABEL_AI, "ChatGPT answer (HC3 test set)")):
            rows = frame[frame["label"] == label]
            preferred = rows[(rows["word_count"] >= 60) & (rows["word_count"] <= 250)]
            rows = preferred if len(preferred) else rows
            if len(rows):
                picked.append((caption, str(rows["text"].iloc[0])))
        return picked

    def report_markdown(self) -> str:
        """Clean and robustness tables, or a hint to run the pipeline."""
        parts = [p.read_text(encoding="utf-8") for p in
                 (self.results_dir / TEST_SETS_MD.name, self.results_dir / CLEAN_METRICS_MD.name,
                  self.results_dir / ROBUSTNESS_MD.name) if p.is_file()]
        if not parts:
            return ("### No results yet\n\nRun the pipeline first: `python run_pipeline.py` "
                    "(or `python -m src.evaluate --all`), then press **Refresh**.")
        return "\n\n---\n\n".join(parts)

    def chart_path(self) -> str | None:
        path = self.results_dir / ROBUSTNESS_CHART_PNG.name
        return str(path) if path.is_file() else None

    # ------------------------------------------------------------------ UI callbacks
    def _on_detect(self, text: str) -> tuple[str, str, str, str]:
        try:
            verdict = self.detect(text)
        except ValueError as error:
            raise gr.Error(str(error)) from None
        except FileNotFoundError:
            raise gr.Error("The trained models were not found. Run `python run_pipeline.py` first.") from None
        note = f"**Note:** {verdict.note}" if verdict.note else ""
        return (verdict_badge_html(verdict), probability_bars_html(verdict, self.zero_shot_name()),
                agreement_text(verdict), note)

    def _on_refresh_report(self):
        chart = self.chart_path()
        return gr.Markdown(value=self.report_markdown()), gr.Image(value=chart, visible=chart is not None)

    # ------------------------------------------------------------------ UI
    def build(self) -> gr.Blocks:
        """Construct the three-tab Gradio interface (does not launch a server)."""
        with gr.Blocks(title=APP_TITLE, analytics_enabled=False) as demo:
            gr.Markdown(f"# {APP_TITLE}\nPaste an English text to estimate whether it was written by a "
                        "person or generated by an AI model such as ChatGPT.")
            with gr.Tab("Detect"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        text = gr.Textbox(label="Text to check", lines=12, max_lines=30,
                                          placeholder="Paste the text here (at least 20 words recommended)...")
                        count = gr.Markdown(word_count_text(""))
                        with gr.Row():
                            clear = gr.Button("Clear", variant="secondary")
                            detect = gr.Button("Detect", variant="primary")
                        examples = self.examples()
                        if examples:
                            gr.Examples(examples=[[t] for _, t in examples], inputs=[text],
                                        example_labels=[c for c, _ in examples],
                                        label="Examples (from the HC3 test set)", cache_examples=False)
                    with gr.Column(scale=2):
                        badge = gr.HTML(label="Verdict")
                        bars = gr.HTML(label="Probability per detector")
                        agreement = gr.Markdown()
                        note = gr.Markdown()
                        gr.Markdown(f"**Limitations:** {LIMITATIONS_NOTE}")
                text.change(word_count_text, inputs=text, outputs=count, queue=False, show_progress="hidden")
                detect.click(self._on_detect, inputs=text, outputs=[badge, bars, agreement, note])
                clear.click(lambda: ("", word_count_text(""), "", "", "", ""), inputs=None,
                            outputs=[text, count, badge, bars, agreement, note], queue=False)
            with gr.Tab("Model report"):
                refresh = gr.Button("Refresh", variant="secondary", size="sm")
                chart = self.chart_path()
                report = gr.Markdown(self.report_markdown())
                image = gr.Image(value=chart, visible=chart is not None, label="Clean accuracy vs accuracy under attack",
                                 type="filepath", interactive=False)
                refresh.click(self._on_refresh_report, inputs=None, outputs=[report, image], queue=False)
            with gr.Tab("About & limitations"):
                gr.Markdown(ABOUT_MARKDOWN)
        return demo

    def launch(self, port: int = APP_PORT, share: bool = False) -> None:
        """Build the UI and serve it on 127.0.0.1 (blocking)."""
        self.build().launch(server_name=APP_HOST, server_port=port, share=share, theme=gr.themes.Soft())


# ---------------------------------------------------------------------- CLI
def _str2bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.app",
        description=f"Launch the '{APP_TITLE}' web app on http://{APP_HOST}:<port>.",
    )
    parser.add_argument("--port", type=int, default=APP_PORT, help="port (default: %(default)s)")
    parser.add_argument("--share", type=_str2bool, default=False,
                        help="create a public Gradio share link: true/false (default: false)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the UI and score one example without starting a server")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="folder with zero_shot.json and distilroberta/ (default: %(default)s)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="folder with the report tables and chart (default: %(default)s)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="folder with test.csv, used for the examples (default: %(default)s)")
    return parser


TIMING_PLACEHOLDER = " ".join(["This sentence is a neutral placeholder used to time the detector."] * 30)


def timing_sample(data_dir: str | Path, target_words: int = 300) -> str:
    """The test text closest to ``target_words`` words (Table 1, Goal 5), or a ~300-word placeholder
    when ``test.csv`` is missing or has no rows."""
    path = Path(data_dir) / "test.csv"
    if path.is_file():
        frame = pd.read_csv(path, encoding="utf-8", dtype={"text": str}, keep_default_na=False)
        if len(frame):
            return str(frame.loc[(frame["word_count"] - target_words).abs().idxmin(), "text"])
    return TIMING_PLACEHOLDER


def dry_run(app: DetectorApp) -> int:
    """Construct the UI, then (if models exist) time one detection of a ~300-word test text."""
    demo = app.build()
    print(f"[app] UI constructed ({type(demo).__name__}, {len(demo.blocks)} blocks); server not started")
    if not app.models_available():
        print("[app] trained models not found - detection check skipped")
        return 0
    sample = timing_sample(app.data_dir)
    started = time.perf_counter()
    app.detect(sample)
    first = time.perf_counter() - started
    started = time.perf_counter()
    verdict = app.detect(sample)
    latency = time.perf_counter() - started
    print(f"[app] example of {word_count(sample)} words: {verdict.label} "
          f"(P_ft={verdict.p_finetuned:.3f}, P_zs={verdict.p_zeroshot:.3f}, agreement={verdict.agreement})")
    print(f"[app] latency: {latency:.2f} s per text (first call incl. model loading: {first:.1f} s)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    configure_console()
    args = build_arg_parser().parse_args(argv)
    app = DetectorApp(models_dir=args.models_dir, results_dir=args.results_dir, data_dir=args.data_dir,
                      device=args.device)
    if args.dry_run:
        return dry_run(app)
    app.launch(port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
