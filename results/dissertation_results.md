# Full results from the dissertation (k = 20)

These numbers were produced by the original dissertation code. RDOS used the simplified neighbourhood from the dissertation (see the main README). Run `python scripts/run_benchmark.py --legacy-rdos` to reproduce them.

| Dataset | Model | Precision | Recall | F1-score | AUC-ROC |
|---|---|---|---|---|---|
| Pima | HDIOD | 0.5806 | 0.2015 | 0.2992 | 0.6912 |
| Pima | RDOS | 0.5690 | 0.2463 | 0.3438 | 0.7319 |
| Pima | LOF | 0.4545 | 0.1119 | 0.1796 | 0.5970 |
| Pima | KNN | 0.5342 | 0.1455 | 0.2287 | 0.7220 |
| Pima | IForest | 0.5776 | 0.2500 | 0.3490 | 0.6923 |
| Vertebral | HDIOD | 0.9524 | 0.1905 | 0.3175 | 0.6810 |
| Vertebral | RDOS | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Vertebral | LOF | 0.8966 | 0.1238 | 0.2176 | 0.5869 |
| Vertebral | KNN | 1.0000 | 0.1429 | 0.2500 | 0.7208 |
| Vertebral | IForest | 0.8936 | 0.2000 | 0.3268 | 0.6985 |
| Thyroid | HDIOD | 0.0977 | 0.9032 | 0.1763 | 0.8958 |
| Thyroid | RDOS | 0.1290 | 0.7849 | 0.2215 | 0.9179 |
| Thyroid | LOF | 0.0819 | 0.3011 | 0.1287 | 0.7220 |
| Thyroid | KNN | 0.2120 | 0.8387 | 0.3384 | 0.9489 |
| Thyroid | IForest | 0.1502 | 0.9140 | 0.2580 | 0.9496 |
| Annthyroid | HDIOD | 0.1919 | 0.5599 | 0.2859 | 0.8067 |
| Annthyroid | RDOS | 0.1361 | 0.2753 | 0.1822 | 0.6411 |
| Annthyroid | LOF | 0.2141 | 0.2622 | 0.2357 | 0.7602 |
| Annthyroid | KNN | 0.2428 | 0.3165 | 0.2748 | 0.7703 |
| Annthyroid | IForest | 0.2000 | 0.4045 | 0.2677 | 0.7817 |
| Arrhythmia | HDIOD | 0.8090 | 0.3478 | 0.4865 | 0.7440 |
| Arrhythmia | RDOS | 0.8529 | 0.2802 | 0.4218 | 0.7393 |
| Arrhythmia | LOF | 0.8974 | 0.1691 | 0.2846 | 0.7232 |
| Arrhythmia | KNN | 0.9773 | 0.2077 | 0.3426 | 0.7443 |
| Arrhythmia | IForest | 0.9118 | 0.2995 | 0.4509 | 0.7424 |
| WPBC | HDIOD | 0.1071 | 0.0638 | 0.0800 | 0.5166 |
| WPBC | RDOS | 0.0667 | 0.0426 | 0.0519 | 0.4675 |
| WPBC | LOF | 0.1053 | 0.0426 | 0.0606 | 0.5175 |
| WPBC | KNN | 0.2000 | 0.0851 | 0.1194 | 0.5273 |
| WPBC | IForest | 0.1333 | 0.0851 | 0.1039 | 0.4816 |

HDIOD used a threshold of median + 1.5 MAD on Arrhythmia and WPBC (2.5 elsewhere). RDOS used bandwidth 0.3 on Arrhythmia (0.1 elsewhere). Isolation Forest used contamination 0.15 and random_state 42.
