# social: 10-fold CV comparison (folds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

## Per-class accuracy (mean ± std over folds)

| class | minirocket | bilstm |
|---|---|---|
| breathingNormally | 0.802 ± 0.045 | 0.861 ± 0.059 |
| coughing | 0.779 ± 0.022 | 0.786 ± 0.028 |
| hyperventilation | 0.744 ± 0.037 | 0.750 ± 0.050 |
| other | 0.746 ± 0.031 | 0.736 ± 0.036 |
| **social (CW3 metric)** | **0.768 ± 0.013** | **0.783 ± 0.016** |
| social uncalibrated | 0.731 | 0.764 |

## social group mean per fold

| fold | minirocket | bilstm |
|---|---|---|
| 0 | 0.753 | 0.772 |
| 1 | 0.769 | 0.752 |
| 2 | 0.771 | 0.785 |
| 3 | 0.786 | 0.808 |
| 4 | 0.756 | 0.791 |
| 5 | 0.756 | 0.790 |
| 6 | 0.792 | 0.796 |
| 7 | 0.767 | 0.762 |
| 8 | 0.776 | 0.789 |
| 9 | 0.753 | 0.788 |

## Pairwise paired tests on the social group mean (n = 10 folds)

| A vs B | mean diff (A−B) | A wins | t-test p | Wilcoxon p |
|---|---|---|---|---|
| minirocket vs bilstm | -0.015 ± 0.017 | 2/10 | 0.0214 | 0.0371 |

## Summed confusion matrices (rows = true, cols = predicted)

**minirocket**

| true \ pred | breathingNor | coughing | hyperventila | other |
|---|---|---|---|---|
| breathingNormally | 9181 | 21 | 164 | 2078 |
| coughing | 5 | 8262 | 1071 | 1271 |
| hyperventilation | 277 | 1060 | 8283 | 1509 |
| other | 2279 | 4662 | 2421 | 27551 |

**bilstm**

| true \ pred | breathingNor | coughing | hyperventila | other |
|---|---|---|---|---|
| breathingNormally | 9857 | 5 | 196 | 1386 |
| coughing | 12 | 8339 | 908 | 1350 |
| hyperventilation | 744 | 438 | 8348 | 1599 |
| other | 6466 | 1786 | 1503 | 27158 |
