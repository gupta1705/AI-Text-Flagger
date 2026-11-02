# Fast-DetectGPT scoring model selection (validation split)

Balanced sample of 300 validation texts (150 human / 150 AI; HC3 and MAGE), measured 15 Oct 2026 on the RTX 2050.
AUROC of the raw Fast-DetectGPT criterion (no calibration). Test sets were not used for this choice.

| Scoring model | Max tokens | Texts/s | Peak GPU memory | AUROC (all) | AUROC (HC3) | AUROC (MAGE) |
|:---|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2.5-0.5B | 256 | 15.7 | 2.10 GB | 0.7136 | 0.9541 | 0.6140 |
| Qwen/Qwen2.5-0.5B | 512 | 8.7 | 2.67 GB | 0.7158 | 0.9565 | 0.6168 |
| Qwen/Qwen2.5-1.5B | 256 | 2.8 | 3.75 GB | 0.6998 | 0.9323 | 0.6054 |
| Qwen/Qwen2.5-1.5B | 512 | 1.7 | 4.33 GB | 0.7043 | 0.9269 | 0.6144 |
| gpt2 | 512 | 16.8 | 1.01 GB | 0.7157 | 0.9911 | 0.5917 |

**Chosen: Qwen/Qwen2.5-0.5B, 256 tokens** - within 0.002 AUROC of the best setting, a modern (2024) scoring model,
and fast enough for the 120,000-variant robustness test. The 1.5B model is no better here, 5x slower, and
nearly fills the 4 GB GPU.
