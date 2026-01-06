# Per-Brand Sentiment Analysis & Data Balancing Report

**Date**: January 6, 2026  
**Status**: ✅ COMPLETED

---

## 📊 Executive Summary

Analisis per-brand untuk 3 coffee shop chain di Indonesia:
- **Kopi Nako**: 1,000 reviews
- **Starbucks**: 583 reviews  
- **Kopi Kenangan**: 1,451 reviews

### Key Findings:

**SEMUA 3 BRANDS MEMILIKI SENTIMENT YANG IMBALANCED** (Majority Positive)

| Brand | Before Balancing | After Balancing | Method |
|-------|-----------------|-----------------|--------|
| **Kopi Nako** | 88.4% pos, 11.6% neg | 50% pos, 50% neg | Oversampling Negative |
| **Starbucks** | 82.7% pos, 17.3% neg | 50% pos, 50% neg | Oversampling Negative |
| **Kopi Kenangan** | 86.8% pos, 13.2% neg | 50% pos, 50% neg | Oversampling Negative |

---

## 🔍 Detailed Analysis per Brand

### 1. **KOPI NAKO** ☕

#### Sentiment Distribution (BEFORE Balancing)
```
Total Reviews: 1,000
├── Positive: 884 (88.4%) ✅
└── Negative: 116 (11.6%) ❌
Balance Ratio: 13.1% ⚠️ IMBALANCED
```

**Analysis**:
- Sangat dominan positive sentiment
- Hanya 13% dari class minority vs class majority
- Menunjukkan kepuasan pelanggan yang tinggi terhadap Kopi Nako
- Tapi DATA SANGAT BIAS untuk negative sentiment

#### Sentiment Distribution (AFTER Balancing)
```
Total Reviews: 1,768 (↑ 76.8% increase)
├── Positive: 884 (50.0%)
└── Negative: 884 (50.0%)
Balance Ratio: 100.0% ✅ BALANCED
```

**Method Applied**: 
- Oversampling negative reviews (116 → 884) menggunakan random resampling dengan replacement
- Hasilnya: perfectly balanced dataset untuk fair model training

---

### 2. **STARBUCKS** ☕

#### Sentiment Distribution (BEFORE Balancing)
```
Total Reviews: 583
├── Positive: 482 (82.7%) ✅
└── Negative: 101 (17.3%) ❌
Balance Ratio: 21.0% ⚠️ IMBALANCED
```

**Analysis**:
- Lebih balanced dibanding Kopi Nako (21% vs 13%)
- Masih dominan positive sentiment
- Negative reviews lebih banyak dibanding Kopi Nako secara persentase
- Menunjukkan beberapa area concern (crowded, expensive, etc)

#### Sentiment Distribution (AFTER Balancing)
```
Total Reviews: 964 (↑ 65.3% increase)
├── Positive: 482 (50.0%)
└── Negative: 482 (50.0%)
Balance Ratio: 100.0% ✅ BALANCED
```

**Method Applied**:
- Oversampling negative reviews (101 → 482)
- Data size naik dari 583 menjadi 964

---

### 3. **KOPI KENANGAN** ☕

#### Sentiment Distribution (BEFORE Balancing)
```
Total Reviews: 1,451
├── Positive: 1,259 (86.8%) ✅
└── Negative: 192 (13.2%) ❌
Balance Ratio: 15.3% ⚠️ IMBALANCED
```

**Analysis**:
- Imbalance ratio mirip dengan Kopi Nako (15.3% vs 13.1%)
- Dataset terbesar dari 3 brands (1,451 reviews)
- Sangat positive-heavy seperti Kopi Nako
- Negative reviews masih relatif sedikit

#### Sentiment Distribution (AFTER Balancing)
```
Total Reviews: 2,518 (↑ 73.6% increase)
├── Positive: 1,259 (50.0%)
└── Negative: 1,259 (50.0%)
Balance Ratio: 100.0% ✅ BALANCED
```

**Method Applied**:
- Oversampling negative reviews (192 → 1,259)
- Largest balanced dataset dari 3 brands

---

## 📈 Comparative Analysis

### Balance Ratio Comparison

```
┌─────────────────┬────────────┬──────────────┬──────────────┐
│ Brand           │ Before (%) │ After (%)    │ Improvement  │
├─────────────────┼────────────┼──────────────┼──────────────┤
│ Kopi Nako       │    13.1%   │   100.0%     │   +863% ⬆️   │
│ Starbucks       │    21.0%   │   100.0%     │   +376% ⬆️   │
│ Kopi Kenangan   │    15.3%   │   100.0%     │   +553% ⬆️   │
└─────────────────┴────────────┴──────────────┴──────────────┘
```

### Size Comparison

```
Brand           Before      After       Increase
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kopi Nako       1,000       1,768       +768
Starbucks         583         964       +381
Kopi Kenangan   1,451       2,518       +1,067
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL           3,034       5,250       +2,216 (+73%)
```

---

## 🛠️ Balancing Method Explanation

### Why Balance?

**Problem**: Imbalanced data causes:
- Models bias toward majority class
- Poor detection of minority class
- Misleading accuracy metrics
- Not representative of reality

**Example**:
- Kopi Nako: If model always predicts "positive", accuracy = 88.4% (but useless!)
- With balancing: Model must learn to detect both classes fairly

### Method Used: **RANDOM OVERSAMPLING**

**Process**:
1. Identify minority class (negative reviews)
2. Randomly resample minority class WITH REPLACEMENT
3. Repeat samples until balance achieved
4. Shuffle combined dataset

**Pros**:
✅ Simple & fast
✅ No data loss
✅ Perfect balance (50-50)
✅ Easy to understand

**Cons**:
⚠️ Duplicated samples (overfitting risk)
⚠️ Not new synthetic data

**Better Alternative (Advanced)**: SMOTE (Synthetic Minority Over-sampling Technique)
- Creates synthetic negative examples
- Reduces overfitting risk
- More computationally expensive

---

## 📁 Generated Files

### Balanced Datasets (Ready for ML Training)
- ✅ `kopi_nako_balanced.csv` (1,768 rows)
- ✅ `starbucks_balanced.csv` (964 rows)
- ✅ `kopi_kenangan_balanced.csv` (2,518 rows)

### Visualizations

| File | Content |
|------|---------|
| `brand_sentiment_before_balancing.png` | 3-panel bar chart showing initial imbalance |
| `brand_sentiment_before_after.png` | 6-panel comparison (before/after for each brand) |
| `brand_sentiment_pie_charts.png` | Pie charts showing percentage distribution |
| `brand_balance_ratio.png` | Line chart comparing balance ratios |

### Summary Report
- ✅ `brand_balance_comparison.csv` (Detailed metrics table)

---

## 💡 Insights & Recommendations

### 1. **Sentiment Reality vs. Data Reality**

**Observed Pattern**: All 3 brands show majority positive sentiment
- ✅ **Reality**: Coffee shop customers are generally satisfied
- ✅ **Good sign**: High satisfaction rates
- ⚠️ **Data concern**: Very few negative examples for learning

### 2. **Brand Comparison**

| Metric | Kopi Nako | Starbucks | Kopi Kenangan |
|--------|-----------|-----------|---------------|
| Positive Sentiment | 88.4% | 82.7% | 86.8% |
| **Ranking** | 🥈 2nd | 🥇 1st (most critical) | 🥉 3rd |
| Negative Reviews | 116 | 101 | 192 |
| **Data Quality** | Sparse negative | Most balanced | Most data |

**Interpretation**:
- **Starbucks**: Most negative-skewed (lowest positive %), suggests more critical customers = GOOD for learning
- **Kopi Nako**: Least negative (highest positive %), most loyal customers but hardest to learn negatives
- **Kopi Kenangan**: Most reviews, but still positive-heavy

### 3. **For ML Model Training**

**USE THE BALANCED DATASETS**:
```python
# ❌ DON'T do this:
df_nako = pd.read_csv('kopinako_main_analysis.csv')  # Imbalanced

# ✅ DO this instead:
df_nako = pd.read_csv('kopi_nako_balanced.csv')  # Balanced
```

**Benefits**:
- Fair model that detects both sentiments
- Better precision/recall balance
- More reliable confidence scores
- Representative feature importance

### 4. **For Production Use**

If you want to train with ORIGINAL IMBALANCED data:
```python
from sklearn.linear_model import LogisticRegression

# Option 1: Class weights
model = LogisticRegression(class_weight='balanced')

# Option 2: Sample weights
sample_weight = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weight)
```

---

## 🎯 Action Items

### Immediate (Use these files)
- [ ] Use balanced CSV files for ML training
- [ ] Generate models using balanced data
- [ ] Evaluate using original test set (imbalanced)

### Short-term (Next week)
- [ ] Retrain all 6 models with balanced data
- [ ] Compare performance: original vs balanced
- [ ] Analyze which features model learned for negative

### Long-term (Continuous improvement)
- [ ] Collect more negative examples
- [ ] Try SMOTE for better synthetic samples
- [ ] Implement real-time model monitoring
- [ ] Gather feedback from model predictions

---

## 📊 Statistical Summary

### Overall Statistics

```
Dataset         Reviews    Positive    Negative    Pos%    Neg%
────────────────────────────────────────────────────────────────
Kopi Nako       1,000      884         116         88.4%   11.6%
Starbucks         583      482         101         82.7%   17.3%
Kopi Kenangan   1,451    1,259         192         86.8%   13.2%
────────────────────────────────────────────────────────────────
COMBINED        3,034    2,625         409         86.5%   13.5%
```

### After Balancing

```
Dataset                Reviews    Positive    Negative    Ratio
────────────────────────────────────────────────────────────────
Kopi Nako Balanced     1,768       884         884        100.0%
Starbucks Balanced       964       482         482        100.0%
Kopi Kenangan Bal.     2,518     1,259       1,259       100.0%
────────────────────────────────────────────────────────────────
TOTAL Balanced         5,250     2,625       2,625       100.0%
```

---

## ✅ Conclusion

### Summary of Findings:

1. **All 3 brands are positive-heavy** (82-88% positive)
   - Reflects real customer satisfaction
   - But creates ML training challenges

2. **All brands have been successfully balanced** using oversampling
   - Balance ratio improved from 13-21% → 100%
   - Total dataset grew from 3,034 → 5,250 samples

3. **Balanced datasets are ready for use**
   - Use for fair model training
   - Will improve minority class detection
   - Enable reliable feature importance analysis

4. **Original imbalanced data should be used for final evaluation**
   - Represents real-world distribution
   - Better reflects actual performance

---

## 📚 References

### Balanced Datasets
- `kopi_nako_balanced.csv` → Use for Kopi Nako-specific analysis
- `starbucks_balanced.csv` → Use for Starbucks-specific analysis
- `kopi_kenangan_balanced.csv` → Use for Kopi Kenangan-specific analysis

### Visualizations
- `brand_sentiment_before_balancing.png` → See initial imbalance
- `brand_sentiment_before_after.png` → Impact of balancing
- `brand_sentiment_pie_charts.png` → Percentage breakdowns
- `brand_balance_ratio.png` → Ratio improvements

### Comparison Data
- `brand_balance_comparison.csv` → Export to Excel for presentation

---

**Report Generated**: January 6, 2026  
**Status**: ✅ Complete & Ready for Use
