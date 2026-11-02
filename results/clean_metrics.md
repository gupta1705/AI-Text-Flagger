# Clean test-set metrics (%)

| Detector | Accuracy | Precision (human) | Precision (AI) | Recall (human) | Recall (AI) | F1 | AUROC |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Log-Likelihood | 88.20 | 99.83 | 80.97 | 76.53 | 99.87 | 89.43 | 99.59 |
| Rank | 70.53 | 93.50 | 63.44 | 44.13 | 96.93 | 76.69 | 91.25 |
| Log-Rank | 87.27 | 99.82 | 79.77 | 74.67 | 99.87 | 88.69 | 99.63 |
| Entropy | 83.27 | 97.16 | 75.70 | 68.53 | 98.00 | 85.42 | 95.04 |
| Fast-DetectGPT | 81.60 | 91.29 | 75.59 | 69.87 | 93.33 | 83.53 | 93.33 |
| DistilRoBERTa | 90.73 | 100.00 | 84.36 | 81.47 | 100.00 | 91.52 | 99.78 |

- Test set: 1,500 texts (750 human / 750 AI), threshold P(AI) >= 0.5.
- F1 treats AI-generated text as the positive class; AUROC uses the raw probabilities.
- Best GPT-2 score on validation F1: **Log-Rank** (the v1 app's zero-shot detector). The v2 web app uses **Fast-DetectGPT**.

## Combined verdict of the web app (%)

| Group | Texts | Coverage | Accuracy (decided) | Labelled AI | Labelled human | Uncertain |
|:---|---:|---:|---:|---:|---:|---:|
| All texts | 1500 | 81.40 | 94.43 | 51.20 | 30.20 | 18.60 |
| Human texts | 750 | 69.47 | 86.95 | 9.07 | 60.40 | 30.53 |
| AI texts | 750 | 93.33 | 100.00 | 93.33 | 0.00 | 6.67 |

- Decision rule of the app (Algorithm 5): DistilRoBERTa and the Fast-DetectGPT detector on the same side of 0.5 -> *Likely AI-generated* / *Likely human-written*, otherwise *Uncertain*. Coverage = share of texts that get a verdict; Accuracy (decided) = accuracy on those texts; Labelled AI = share of the group labelled *Likely AI-generated* (for human texts: the false-accusation rate).
