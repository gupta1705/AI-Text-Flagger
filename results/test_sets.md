# Results on all test sets (v2)

## Accuracy (%)

| Detector | HC3 test | MAGE test (in-distribution) | MAGE OOD (unseen domains, GPT-4) | Modern AI (unseen 2025-26 models) |
|:---|---:|---:|---:|---:|
| DistilRoBERTa v1 (HC3 only) | 97.73 | 63.15 | 76.06 | 54.85 |
| Log-Likelihood v1 (HC3 calibration) | 97.87 | 63.25 | 67.31 | 50.50 |
| Log-Likelihood | 88.20 | 63.80 | 70.14 | 58.19 |
| Rank | 70.53 | 61.35 | 65.51 | 68.90 |
| Log-Rank | 87.27 | 64.40 | 69.88 | 55.85 |
| Entropy | 83.27 | 62.60 | 61.52 | 49.16 |
| Fast-DetectGPT | 81.60 | 57.90 | 68.47 | 66.22 |
| DistilRoBERTa v2 (HC3 + MAGE) | 90.73 | 85.10 | 75.93 | 70.23 |

## AUROC (%)

| Detector | HC3 test | MAGE test (in-distribution) | MAGE OOD (unseen domains, GPT-4) | Modern AI (unseen 2025-26 models) |
|:---|---:|---:|---:|---:|
| DistilRoBERTa v1 (HC3 only) | 99.96 | 66.89 | 85.38 | 64.80 |
| Log-Likelihood v1 (HC3 calibration) | 99.59 | 66.42 | 78.26 | 64.03 |
| Log-Likelihood | 99.59 | 66.42 | 78.26 | 64.03 |
| Rank | 91.25 | 67.13 | 72.13 | 75.50 |
| Log-Rank | 99.63 | 68.00 | 77.63 | 59.93 |
| Entropy | 95.04 | 65.32 | 62.72 | 50.67 |
| Fast-DetectGPT | 93.33 | 60.26 | 75.41 | 73.55 |
| DistilRoBERTa v2 (HC3 + MAGE) | 99.78 | 94.68 | 90.26 | 77.99 |

## Web app verdict before vs after (%)

| Test set | Version | Coverage | Accuracy (decided) | AI texts caught | Human texts wrongly flagged |
|:---|---:|---:|---:|---:|---:|
| HC3 test | before (v1) | 96.53 | 99.52 | 98.00 | 0.93 |
| HC3 test | after (v2) | 81.40 | 94.43 | 93.33 | 9.07 |
| MAGE test (in-distribution) | before (v1) | 83.00 | 65.90 | 27.00 | 1.70 |
| MAGE test (in-distribution) | after (v2) | 59.70 | 86.01 | 48.90 | 12.30 |
| MAGE OOD (unseen domains, GPT-4) | before (v1) | 74.77 | 79.00 | 53.47 | 15.51 |
| MAGE OOD (unseen domains, GPT-4) | after (v2) | 67.31 | 82.98 | 72.01 | 23.13 |
| Modern AI (unseen 2025-26 models) | before (v1) | 87.63 | 53.05 | 4.03 | 3.33 |
| Modern AI (unseen 2025-26 models) | after (v2) | 65.89 | 77.66 | 39.60 | 9.33 |

- v1 = DistilRoBERTa trained on HC3 only + GPT-2 log-likelihood; v2 = DistilRoBERTa trained on HC3 + MAGE + Fast-DetectGPT. Both apply the same agreement rule (Algorithm 5).
- AI texts caught = share of AI texts labelled *Likely AI-generated*; Human texts wrongly flagged = share of human texts labelled *Likely AI-generated*; the rest are *Uncertain* or labelled human.

- Test sets: HC3 test = `data/test.csv`; MAGE test (in-distribution) = `data/test_mage.csv`; MAGE OOD (unseen domains, GPT-4) = `data/test_ood.csv`; Modern AI (unseen 2025-26 models) = `data/test_modern.csv`.
- MAGE OOD contains four domains absent from training (CNN news, IMDB reviews, PubMed abstracts, DialogSum summaries) with texts from GPT-4, an unseen generator; Modern AI pairs human MAGE test texts with same-topic, same-length texts written by qwen/qwen3.8-27b and openai/gpt-oss-20b (2025-26).
