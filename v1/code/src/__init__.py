"""AI-Text-Flagger: zero-shot statistical and fine-tuned transformer detectors for AI-generated text.

CSET312 mini-project, Bennett University. The package follows the module layout of the submitted
report (Section 3.6): ``data`` (HC3DataModule), ``zero_shot`` (ZeroShotDetector), ``finetune``
(FineTunedDetector), ``perturb`` (Perturber), ``evaluate`` (Evaluator) and ``app`` (DetectorApp),
with shared constants in ``config`` and text normalisation in ``text_utils``.

Run every module from the repository root, e.g. ``python -m src.data --help``.
"""

__version__ = "1.0.0"
