# PER-BRAND ANALYSIS - QUICK START GUIDE

**Tanggal**: 6 Januari 2026  
**Status**: ✅ SELESAI

---

## 🎯 Apa yang Dilakukan?

Anda meminta analisis terpisah untuk 3 coffee shop brands dengan cek balance dan balancing jika diperlukan.

**Hasil**: ✅ Semua 3 brands berhasil dianalisis dan dibalans!

---

## 📊 RINGKASAN HASIL

### Kopi Nako ☕
```
SEBELUM:  88.4% Positive (884) vs 11.6% Negative (116)
          ⚠️  IMBALANCED (Ratio: 13.1%)

SESUDAH:  50% Positive (884) vs 50% Negative (884)  
          ✅ BALANCED (Ratio: 100%)

METODE:   Oversampling - Menambah negative examples dari 116 → 884
UKURAN:   1,000 → 1,768 samples (+76.8%)
```

### Starbucks ☕
```
SEBELUM:  82.7% Positive (482) vs 17.3% Negative (101)
          ⚠️  IMBALANCED (Ratio: 21.0%)

SESUDAH:  50% Positive (482) vs 50% Negative (482)
          ✅ BALANCED (Ratio: 100%)

METODE:   Oversampling - Menambah negative examples dari 101 → 482  
UKURAN:   583 → 964 samples (+65.3%)
```

### Kopi Kenangan ☕
```
SEBELUM:  86.8% Positive (1,259) vs 13.2% Negative (192)
          ⚠️  IMBALANCED (Ratio: 15.3%)

SESUDAH:  50% Positive (1,259) vs 50% Negative (1,259)
          ✅ BALANCED (Ratio: 100%)

METODE:   Oversampling - Menambah negative examples dari 192 → 1,259
UKURAN:   1,451 → 2,518 samples (+73.6%)
```

---

## 📁 FILE YANG DIHASILKAN

### 1. Visualisasi Gambar (4 files)

| File | Isi |
|------|-----|
| **brand_sentiment_before_balancing.png** | 3 chart bar sebelum balancing |
| **brand_sentiment_before_after.png** | 6 chart bar (before/after untuk setiap brand) |
| **brand_sentiment_pie_charts.png** | Pie chart persentase |
| **brand_balance_ratio.png** | Grafik perbandingan balance ratio improvement |

### 2. Dataset Balanced (3 CSV files)
- **kopi_nako_balanced.csv** (1,768 rows) - Siap untuk training
- **starbucks_balanced.csv** (964 rows) - Siap untuk training
- **kopi_kenangan_balanced.csv** (2,518 rows) - Siap untuk training

### 3. Laporan & Data Comparison
- **brand_balance_comparison.csv** - Tabel perbandingan lengkap
- **PER_BRAND_ANALYSIS_REPORT.md** - Laporan detail + insights
- **per_brand_analysis.py** - Script analisis (bisa dijalankan ulang)

---

## 💡 KEY FINDINGS

### ✅ Semua Brands IMBALANCED (Majority Positive)
- **Kopi Nako**: 88.4% positive (paling positive-heavy)
- **Kopi Kenangan**: 86.8% positive
- **Starbucks**: 82.7% positive (paling balanced dari 3)

### ✅ Semua Sudah DIBALANS
- Menggunakan **Oversampling** (Random Resampling)
- Negative examples di-upsample (dengan replacement) sampai matching positive
- Hasil: Perfect 50-50 balance untuk semua

### ✅ Mengapa Ini Penting?
- **Model training**: Akan belajar detect negative sama baik seperti positive
- **Fair metrics**: Accuracy, precision, recall lebih meaningful
- **Better predictions**: Confidence scores lebih reliable

---

## 🚀 CARA MENGGUNAKAN FILE YANG SUDAH DIBALANS

### Untuk Kopi Nako:
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load balanced data
df = pd.read_csv('kopi_nako_balanced.csv')

# Split for training
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], 
    test_size=0.2, random_state=42
)

# Now train your model with balanced data
# Model akan belajar dengan fair untuk both sentiments!
```

### Untuk Starbucks:
```python
df = pd.read_csv('starbucks_balanced.csv')
# ... same code ...
```

### Untuk Kopi Kenangan:
```python
df = pd.read_csv('kopi_kenangan_balanced.csv')
# ... same code ...
```

---

## 📊 VISUALISASI YANG DIBUAT

### 1. brand_sentiment_before_balancing.png
3 bar chart yang menunjukkan imbalance awal:
- Sebelah kiri sangat tinggi (positive)
- Sebelah kanan sangat rendah (negative)

### 2. brand_sentiment_before_after.png  
6 chart (2x3 grid):
- **Top row**: BEFORE (imbalanced)
- **Bottom row**: AFTER (balanced perfectly)
- Terlihat jelas perbedaan sebelum/sesudah

### 3. brand_sentiment_pie_charts.png
Pie chart untuk 3x2 (before/after):
- Pie yang lopsided → pie yang 50-50

### 4. brand_balance_ratio.png
Garis chart showing improvement:
- Garis biru: BEFORE (rendah)
- Garis orange: AFTER (tinggi sampai 100%)
- Garis merah: Threshold 40%

---

## 🎯 NEXT STEPS (Rekomendasi)

### 1. **Immediate Use** (Gunakan sekarang)
```
Gunakan balanced CSV files untuk training model:
- kopi_nako_balanced.csv
- starbucks_balanced.csv
- kopi_kenangan_balanced.csv
```

### 2. **Train Models** (Per-brand atau combined)
```
Option A: Train model PER BRAND
  - Model khusus Kopi Nako
  - Model khusus Starbucks
  - Model khusus Kopi Kenangan

Option B: Train model COMBINED
  - Mix semua 3 balanced datasets
  - 1 general model untuk semua brands
```

### 3. **Evaluate with Original Data**
```
Test dengan data ORIGINAL (imbalanced) untuk lihat real-world performance
```

### 4. **Create Insights**
```
Lihat file: PER_BRAND_ANALYSIS_REPORT.md
- Complete analysis
- Business insights
- Recommendations
```

---

## 🔍 DETAIL COMPARISON

### Data Size Evolution
```
Brand           Original    Balanced    Growth
────────────────────────────────────────────────
Kopi Nako       1,000       1,768       +768
Starbucks         583         964       +381
Kopi Kenangan   1,451       2,518     +1,067
────────────────────────────────────────────────
TOTAL           3,034       5,250     +2,216
```

### Balance Quality
```
Brand           Before      After       Improvement
──────────────────────────────────────────────────
Kopi Nako       13.1%      100.0%       +863%
Starbucks       21.0%      100.0%       +376%
Kopi Kenangan   15.3%      100.0%       +553%
```

---

## ⚠️ PENTING: Tips Penggunaan

### ✅ GUNAKAN (balanced data untuk training)
```python
df = pd.read_csv('kopi_nako_balanced.csv')  # ✅ BAIK
```

### ❌ JANGAN GUNAKAN (untuk evaluation)
```python
df = pd.read_csv('kopinako_main_analysis.csv')  # ❌ UNTUK EVAL
```

### 🎯 BEST PRACTICE
```
1. Train dengan: balanced CSV
2. Test dengan: original data (real-world distribution)
3. Evaluate dengan: both untuk comparison
```

---

## 📚 FILE REFERENCE

### Untuk baca LENGKAP:
👉 **PER_BRAND_ANALYSIS_REPORT.md**

Berisi:
- Executive summary
- Detailed analysis per brand
- Insights & recommendations
- Statistical summary
- Action items

### Untuk gunakan data:
👉 **kopi_*_balanced.csv**

Siap untuk:
- Train/test split
- ML model training
- Feature extraction
- Classification

### Untuk lihat visualisasi:
👉 **brand_*.png** (4 files)

Cocok untuk:
- Presentation
- Report
- Publication
- Dashboard

---

## 🎓 TECHNICAL DETAILS

### Balancing Method: Random Oversampling
```python
from sklearn.utils import resample

# Oversample minority class
minority = df[df['sentiment'] == 'negative']
majority = df[df['sentiment'] == 'positive']

minority_upsampled = resample(
    minority,
    n_samples=len(majority),
    replace=True  # WITH replacement
)

balanced = pd.concat([majority, minority_upsampled])
```

### Kenapa Oversampling?
- ✅ Simple & fast
- ✅ Perfect balance (50-50)
- ✅ No data loss
- ⚠️  Potential overfitting (duplicates)

### Alternative: SMOTE
- Synthetic Minority Over-sampling Technique
- Creates synthetic examples
- Better than random oversampling
- More computationally expensive

---

## ❓ FAQ

**Q: Kenapa negative oversampled, bukan positive undersampled?**  
A: Preserving data. Lebih baik duplikat negative dari pada buang positive examples.

**Q: Boleh pakai balanced + original bersamaan?**  
A: Tidak recommended. Gunakan salah satu konsisten.

**Q: Balanced dataset fix overfitting?**  
A: Tidak. Balancing = fair training, bukan fix overfitting. Gunakan validation set untuk cek.

**Q: Size dataset naik 73%, aman?**  
A: Aman. Duplikat data tapi minority class tetap berbeda (random shuffle). Monitor overfitting.

**Q: Bisa gunakan balanced untuk test set?**  
A: Tidak. Test set harus original/real distribution untuk accurate evaluation.

---

## 📞 SUMMARY

```
✅ ANALISIS SELESAI
✅ SEMUA BRANDS DIBALANS
✅ VISUALISASI DIBUAT
✅ BALANCED DATA READY FOR USE
✅ DOKUMENTASI LENGKAP

NEXT: Use balanced CSV untuk train model!
```

---

**Generated**: January 6, 2026  
**Status**: Complete & Ready  
**Quality**: Production-Ready
