# PHASE 1 — Data Summary

This document reports the exploratory data analysis and preprocessing decisions for fine-tuning a BERT-based classifier on the `shreyaspullehf/emotion_dataset_100k` dataset.

---

## 1. Dataset Overview

- **HuggingFace Dataset ID:** `shreyaspullehf/emotion_dataset_100k`
- **Original Train Samples:** 99,746
- **Number of Emotion Classes:** 10

---

## 2. Token Length Analysis (BERT Tokenization)

Tokenization performed using `bert-base-uncased`.

| Metric | Value |
|--------|--------|
| Mean Token Length | 41.11 |
| Median Token Length | 39 |
| Maximum Token Length | 155 |
| 95th Percentile | 65 |

---

## 3. MAX_LENGTH Selection Strategy

A percentile-based truncation strategy was applied:

- **Percentile Used:** 95%
- **Raw Percentile Value:** 65
- **Rounded to Nearest Multiple of 8:** 72
- **Final MAX_LENGTH:** 72
- **Truncation Ratio:** 0.0342 (3.42% of samples truncated)

### Justification

Using the 95th percentile ensures:

- Minimal information loss (only 3.42% truncated)
- Efficient memory utilization
- Stable batch computation
- Reduced padding overhead

---

## 4. Label Distribution

| Label | Count | Proportion |
|--------|--------|------------|
| disgust | 9975 | 0.1000 |
| drive | 9996 | 0.1002 |
| embarrassment | 9969 | 0.0999 |
| excitement | 9947 | 0.0997 |
| fear | 9978 | 0.1000 |
| happiness | 9995 | 0.1002 |
| loneliness | 9992 | 0.1002 |
| love | 9980 | 0.1001 |
| sadness | 9989 | 0.1001 |
| surprise | 9925 | 0.0995 |

---

## 5. Imbalance Analysis

- **Majority Class Count:** 9996  
- **Minority Class Count:** 9925  
- **Imbalance Ratio (Majority/Minority):** 1.01  

The dataset is nearly perfectly balanced. The negligible imbalance ratio (1.01) indicates that no class weighting or resampling strategy is required.

---

## 6. Validation Split Strategy

A **stratified 90/10 split** was applied using:


datasets.train_test_split(stratify_by_column='label', seed=42)
Validation Fraction: 10% 

Maximum Proportion Difference (train vs val): 0.0001

This confirms statistically consistent label distributions between training and validation sets.

## 7. Reproducibility

All splits were generated using:

Random Seed: 42

Deterministic HuggingFace dataset operations

This ensures identical splits across multiple runs.

## 8. Generated Plots

reports/figures/label_distribution.png

reports/figures/text_length_distribution.png