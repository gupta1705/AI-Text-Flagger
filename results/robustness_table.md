# Robustness to character-level perturbations

## Accuracy under attack, AUA (%) - clean prediction and all k = 10 variants correct

| Detector | Normal | Swap 1 | Swap 2 | Drop 1 | Drop 2 | Add 1 | Add 2 | Key 1 | Key 2 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Log-Likelihood | 97.87 | 94.40 | 89.33 | 95.13 | 90.87 | 94.20 | 88.47 | 94.47 | 88.80 |
| Rank | 81.00 | 45.73 | 38.13 | 48.33 | 41.07 | 40.60 | 36.60 | 41.13 | 37.60 |
| Log-Rank | 97.47 | 95.27 | 90.00 | 95.60 | 91.67 | 94.67 | 88.67 | 94.87 | 89.20 |
| Entropy | 87.73 | 82.60 | 78.60 | 83.13 | 79.67 | 82.53 | 78.07 | 82.07 | 77.47 |
| DistilRoBERTa | 97.73 | 97.33 | 97.20 | 97.33 | 97.20 | 97.40 | 97.00 | 97.27 | 97.07 |

## Mean per-variant accuracy (%)

| Detector | Normal | Swap 1 | Swap 2 | Drop 1 | Drop 2 | Add 1 | Add 2 | Key 1 | Key 2 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Log-Likelihood | 97.87 | 96.62 | 93.13 | 96.93 | 93.97 | 96.30 | 92.07 | 96.47 | 92.40 |
| Rank | 81.00 | 79.03 | 71.67 | 80.06 | 74.87 | 75.30 | 65.39 | 76.97 | 67.96 |
| Log-Rank | 97.47 | 97.03 | 93.94 | 97.13 | 94.98 | 96.74 | 93.11 | 96.88 | 93.49 |
| Entropy | 87.73 | 87.27 | 86.53 | 87.31 | 86.69 | 87.35 | 86.45 | 87.17 | 86.21 |
| DistilRoBERTa | 97.73 | 98.46 | 98.83 | 98.38 | 98.82 | 98.59 | 98.89 | 98.47 | 98.85 |

- Test set: 1,500 texts (750 human / 750 AI), threshold P(AI) >= 0.5.
- AUA = accuracy under attack: a test text counts as correct only if all k = 10 perturbed variants are classified correctly; a text whose clean prediction is already wrong counts as not robust (the original is treated as the unedited variant), so AUA never exceeds Normal (= clean accuracy). The mean per-variant accuracy ignores the clean prediction.
- GPT-2 reads only the first 512 tokens. 33 of 1,500 test texts (33 human / 0 AI) are longer; in 548 of their 2,640 variants no edit falls inside that window, so the four zero-shot detectors score the clean text there (DistilRoBERTa reads up to 8 x 256 tokens). This can raise the zero-shot AUA by at most 2.20 percentage points (column `edit_in_gpt2_window` of robustness_variants.csv.gz).
- Operations (Pruthi et al., 2019): Swap = exchange two neighbouring letters, Drop = delete a letter, Add = insert a random letter, Key = replace a letter with a neighbouring QWERTY key; 1 / 2 = number of edited words (one edit per word, first letter kept).
