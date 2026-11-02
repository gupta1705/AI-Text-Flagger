"""Project-wide constants: paths, data sizes, model settings and evaluation settings.

Every tunable value of the pipeline lives here so that the code, the README and the submitted
report (Section 3.3, Algorithms 1-5) cannot drift apart. Modules import these names instead of
hard-coding numbers; command-line flags override them for a single run only.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------------
# Paths (all generated artefacts stay inside the repository and are git-ignored where large)
# --------------------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
HC3_DEFAULT_PATH = RAW_DIR / "all.jsonl"
HC3_CACHED_PATH = Path.home() / ".cache" / "hc3" / "all.jsonl"

SPLIT_NAMES = ("train", "val", "test")
SUMMARY_FILENAME = "summary.json"

ZERO_SHOT_JSON = MODELS_DIR / "zero_shot.json"
FINETUNED_DIR = MODELS_DIR / "distilroberta"
TRAINING_LOG_FILENAME = "training_log.json"

CLEAN_METRICS_CSV = RESULTS_DIR / "clean_metrics.csv"
CLEAN_METRICS_MD = RESULTS_DIR / "clean_metrics.md"
CLEAN_PREDICTIONS_CSV = RESULTS_DIR / "clean_predictions.csv"
VERDICT_METRICS_CSV = RESULTS_DIR / "verdict_metrics.csv"
ROBUSTNESS_CSV = RESULTS_DIR / "robustness.csv"
ROBUSTNESS_MD = RESULTS_DIR / "robustness_table.md"
ROBUSTNESS_SAMPLES_CSV = RESULTS_DIR / "robustness_per_sample.csv"
ROBUSTNESS_VARIANTS_CSV = RESULTS_DIR / "robustness_variants.csv.gz"
ROBUSTNESS_CHART_PNG = RESULTS_DIR / "robustness_chart.png"
ROBUST_STATS_CACHE = CACHE_DIR / "gpt2_stats_robust.csv"

# --------------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------------------------
# Data (report Section 3.3, steps 1-3)
# --------------------------------------------------------------------------------------------
LABEL_HUMAN = 0
LABEL_AI = 1
CLASS_NAMES = {LABEL_HUMAN: "human", LABEL_AI: "ai"}

MIN_WORDS = 20            # answers with fewer words are removed
N_FIT = 4_500             # training texts used for fitting (balanced 50/50)
N_VAL = 500               # validation texts (balanced 50/50); N_FIT + N_VAL = 5,000 "training" texts
N_TEST = 1_500            # held-out test texts (750 human / 750 AI)
POOL_FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}   # share of questions per split pool

# --------------------------------------------------------------------------------------------
# v2: MAGE training data and extra test sets (src/mage.py, src/modern_set.py)
# --------------------------------------------------------------------------------------------
MAGE_URL = "https://huggingface.co/datasets/yaful/MAGE/resolve/main"
MAGE_RAW_DIR = RAW_DIR / "mage"
MAGE_CACHED_DIR = Path.home() / ".cache" / "mage"
MAGE_FILES = {"train": "train.csv", "valid": "valid.csv", "test": "test.csv", "ood": "test_ood_set_gpt.csv"}
MAGE_N_TRAIN = 9_000      # added to the 4,500 HC3 training texts (balanced, 10 domains)
MAGE_N_VAL = 1_000        # added to the 500 HC3 validation texts
MAGE_N_TEST = 2_000       # in-distribution MAGE test sample (data/test_mage.csv)
# Test sets evaluated by src.evaluate: split file -> display name. "test" is the HC3 test split of v1.
TEST_SETS = {
    "test": "HC3 test",
    "test_mage": "MAGE test (in-distribution)",
    "test_ood": "MAGE OOD (unseen domains, GPT-4)",
    "test_modern": "Modern AI (unseen 2025-26 models)",
}

# --------------------------------------------------------------------------------------------
# Zero-shot detector (Algorithms 1-2, Listing 2)
# --------------------------------------------------------------------------------------------
ZS_MODEL_NAME = "gpt2"
ZS_MAX_TOKENS = 512
ZS_BATCH_SIZE = 8
ZS_METRICS = ("loglik", "rank", "logrank", "entropy")
ZS_CACHE_CHUNK = 1_024    # texts scored between two appends to the statistics cache

# --------------------------------------------------------------------------------------------
# v2: Fast-DetectGPT zero-shot detector (src/fast_detect.py; Bao et al., ICLR 2024)
# --------------------------------------------------------------------------------------------
FD_KEY = "fastdetect"
FD_MODEL_NAME = "Qwen/Qwen2.5-0.5B"   # scoring model (2024 base LM); chosen on the validation split
# (results/fast_detect_model_selection.md: 0.5B matched 1.5B and GPT-2 in validation AUROC at 5x the
# speed of 1.5B, which also nearly fills the 4 GB GPU)
FD_MAX_TOKENS = 256
FD_BATCH_SIZE = 8
FAST_DETECT_JSON = MODELS_DIR / "fast_detect.json"

# --------------------------------------------------------------------------------------------
# Fine-tuned detector (Algorithm 3)
# --------------------------------------------------------------------------------------------
FT_MODEL_NAME = "distilroberta-base"
FT_MAX_LENGTH = 256       # tokens per training sequence / inference chunk (incl. <s> and </s>)
FT_MAX_CHUNKS = 8         # inference: at most 8 consecutive 256-token chunks per text
FT_EPOCHS = 3
FT_BATCH_SIZE = 8
FT_LEARNING_RATE = 2e-5
FT_WEIGHT_DECAY = 0.01
FT_WARMUP_RATIO = 0.1
FT_GRAD_ACCUM = 1
# Addition to Algorithm 3: clip the gradient norm at 1.0 before each AdamW step (the Hugging Face
# Trainer default, a guard against fp16 loss spikes). 0 disables it (--grad-clip 0).
FT_GRAD_CLIP_NORM = 1.0

# --------------------------------------------------------------------------------------------
# Perturbations and robustness (Algorithm 4, report Section 3.3 step 7)
# --------------------------------------------------------------------------------------------
PERTURB_OPS = ("swap", "drop", "key", "add")
PERTURB_LEVELS = (1, 2)
K_VARIANTS = 10
# Column order of the robustness table (survey Table 5 layout).
ROBUSTNESS_CONDITIONS = (
    ("swap", 1), ("swap", 2), ("drop", 1), ("drop", 2),
    ("add", 1), ("add", 2), ("key", 1), ("key", 2),
)
OP_DISPLAY = {"swap": "Swap", "drop": "Drop", "add": "Add", "key": "Key"}

# --------------------------------------------------------------------------------------------
# Detectors and decision rule (Algorithm 5, Listing 4)
# --------------------------------------------------------------------------------------------
FINETUNED_KEY = "distilroberta"
DETECTORS = (*ZS_METRICS, FD_KEY, FINETUNED_KEY)
DETECTOR_DISPLAY = {
    "loglik": "Log-Likelihood",
    "rank": "Rank",
    "logrank": "Log-Rank",
    "entropy": "Entropy",
    FD_KEY: "Fast-DetectGPT",
    FINETUNED_KEY: "DistilRoBERTa",
}
# v2: the web app combines DistilRoBERTa with Fast-DetectGPT (v1 used the GPT-2 log-likelihood score).
APP_ZERO_SHOT = FD_KEY
THRESHOLD = 0.5
SHORT_TEXT_WORDS = 20
VERDICT_AI = "Likely AI-generated"
VERDICT_HUMAN = "Likely human-written"
VERDICT_UNCERTAIN = "Uncertain – review manually"

# --------------------------------------------------------------------------------------------
# Smoke run (run_pipeline.py --smoke)
# --------------------------------------------------------------------------------------------
SMOKE_LIMIT = 64
SMOKE_EPOCHS = 1
SMOKE_K = 2
# The smoke run writes its own data/, models/ and results/ here (git-ignored), so it never
# replaces the full-size outputs in DATA_DIR, MODELS_DIR and RESULTS_DIR.
SMOKE_DIR = ROOT_DIR / "smoke"

# --------------------------------------------------------------------------------------------
# v1 (HC3-only) models kept for the before/after comparison (src/evaluate.py --sets)
# --------------------------------------------------------------------------------------------
V1_DIR = ROOT_DIR / "v1"
V1_MODELS_DIR = V1_DIR / "models"
V1_ZERO_SHOT_METRIC = "loglik"     # the zero-shot score the v1 app used
TEST_SET_METRICS_CSV = RESULTS_DIR / "test_sets_metrics.csv"
TEST_SET_VERDICTS_CSV = RESULTS_DIR / "test_sets_verdicts.csv"
TEST_SETS_MD = RESULTS_DIR / "test_sets.md"

# --------------------------------------------------------------------------------------------
# Web application
# --------------------------------------------------------------------------------------------
APP_TITLE = "AI Text Detector"
APP_HOST = "127.0.0.1"
APP_PORT = 7860
