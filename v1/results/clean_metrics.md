# Clean test-set metrics (%)

| Detector | Accuracy | Precision (human) | Precision (AI) | Recall (human) | Recall (AI) | F1 | AUROC |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Log-Likelihood | 97.87 | 97.99 | 97.74 | 97.73 | 98.00 | 97.87 | 99.59 |
| Rank | 81.00 | 86.50 | 76.94 | 73.47 | 88.53 | 82.33 | 91.25 |
| Log-Rank | 97.47 | 97.59 | 97.34 | 97.33 | 97.60 | 97.47 | 99.63 |
| Entropy | 87.73 | 89.09 | 86.47 | 86.00 | 89.47 | 87.94 | 95.04 |
| DistilRoBERTa | 97.73 | 100.00 | 95.66 | 95.47 | 100.00 | 97.78 | 99.96 |

- Test set: 1,500 texts (750 human / 750 AI), threshold P(AI) >= 0.5.
- F1 treats AI-generated text as the positive class; AUROC uses the raw probabilities.
- Zero-shot score used by the web app: **Log-Likelihood** (best validation F1).

## Combined verdict of the web app (%)

| Group | Texts | Coverage | Accuracy (decided) | Labelled AI | Labelled human | Uncertain |
|:---|---:|---:|---:|---:|---:|---:|
| All texts | 1500 | 96.53 | 99.52 | 49.47 | 47.07 | 3.47 |
| Human texts | 750 | 95.07 | 99.02 | 0.93 | 94.13 | 4.93 |
| AI texts | 750 | 98.00 | 100.00 | 98.00 | 0.00 | 2.00 |

- Decision rule of the app (Algorithm 5): DistilRoBERTa and the Log-Likelihood detector on the same side of 0.5 -> *Likely AI-generated* / *Likely human-written*, otherwise *Uncertain*. Coverage = share of texts that get a verdict; Accuracy (decided) = accuracy on those texts; Labelled AI = share of the group labelled *Likely AI-generated* (for human texts: the false-accusation rate).
