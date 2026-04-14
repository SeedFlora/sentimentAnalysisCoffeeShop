# Revisi Reviewer yang Sudah Dilakukan

Dokumen ini merangkum perbaikan yang sudah dilakukan pada project berdasarkan komentar reviewer. Fokus revisi mencakup metodologi, eksperimen model, baseline comparison, evaluasi, ABSA pipeline, Table 7, dan penambahan IndoBERT sebagai model pembanding berbasis transformer.

## Ringkasan Status

| Area Revisi | Status | Output Utama |
|---|---|---|
| Pipeline 3 brand | Selesai | `revised_outputs/revised_clean_reviews.csv` |
| ABSA aspect extraction | Selesai | `revised_outputs/revised_absa_aspect_level.csv` |
| Aspect distribution Table 2-4 | Selesai | `revised_outputs/revised_aspect_distribution_table.csv` |
| Baseline model comparison | Selesai | `revised_outputs/revised_model_holdout_metrics.csv` |
| 5-fold cross-validation | Selesai | `revised_outputs/revised_model_cv_summary.csv` |
| Per-class evaluation | Selesai | `revised_outputs/revised_classification_report.csv` |
| Confusion matrix | Selesai | `revised_outputs/revised_confusion_matrices.csv` |
| Table 7 neutral word removal | Selesai | `revised_outputs/revised_sentiment_word_frequency.csv` |
| IndoBERT experiment | Selesai | `revised_outputs/revised_indobert_holdout_metrics.csv` |
| Dashboard bug fix | Selesai | `streamlit_app.py` |
| Paper/PDF text revision | Belum langsung diedit | Perlu file Word/LaTeX sumber |

## 1. Brand Name dan Alias

### Komentar Reviewer

Reviewer menyarankan agar brand name seperti Starbucks, Kopi Kenangan, dan Kopi Nako diganti menjadi alias jika tidak memiliki izin penggunaan brand.

### Perbaikan yang Sudah Dilakukan

Pada pipeline revisi, brand asli tetap disimpan untuk kebutuhan data internal, tetapi sudah ditambahkan kolom alias:

| Brand Asli | Alias Akademik |
|---|---|
| Starbucks | Global Brand |
| Kopi Kenangan | National Brand |
| Kopi Nako | Local Brand |

### File Terkait

- `revised_outputs/revised_brand_alias_mapping.csv`
- `revised_outputs/revised_clean_reviews.csv`

### Catatan untuk Manuskrip

Jika editor meminta anonymization penuh, gunakan alias Global Brand, National Brand, dan Local Brand di seluruh naskah. Jika brand asli tetap dipakai, tambahkan academic disclaimer bahwa studi ini tidak berafiliasi, tidak disponsori, dan tidak mewakili endorsement dari brand mana pun.

## 2. Table 7: Kata “Coffee” Dihapus

### Komentar Reviewer

Reviewer menyatakan bahwa kata “coffee” muncul pada konotasi positif dan negatif, padahal kata tersebut netral dan tidak dapat dikategorikan sebagai sentimen positif atau negatif.

### Perbaikan yang Sudah Dilakukan

Frequency analysis sudah direvisi. Kata netral/domain nouns dihapus dari tabel frekuensi, termasuk:

- `coffee`
- `kopi`
- `starbucks`
- `nako`
- `kenangan`
- `tempat`
- `place`
- filler words seperti `nya` dan `tapi`

Selain itu, frequency table sekarang memakai daftar istilah sentiment-bearing sehingga hasilnya lebih relevan untuk analisis sentimen.

### Output Baru

- `revised_outputs/revised_sentiment_word_frequency.csv`

### Contoh Hasil Revisi untuk Kopi Nako

| Sentiment | Top Terms |
|---|---|
| Positive | delicious, cozy, good, friendly, favorite, spacious, clean, worth |
| Negative | less, expensive, slow, bad, inconsistent, not delicious, not friendly, disappointed |

## 3. SDG Statement

### Komentar Reviewer

Reviewer menyatakan bahwa SDG tidak dibahas dalam paper, sehingga pernyataan SDG sebaiknya dihapus.

### Status Perbaikan

Bagian ini merupakan revisi naskah, bukan coding. Karena file sumber Word/LaTeX belum tersedia, PDF belum bisa diedit langsung.

### Rekomendasi Final

Hapus pernyataan SDG dari abstract/conclusion agar tidak membuka risiko komentar lanjutan. Jika ingin tetap dipertahankan, perlu tambahan pembahasan khusus mengenai hubungan framework Python, digital infrastructure, SME competitiveness, dan SDG 9.

## 4. Limitations dan Future Research

### Komentar Reviewer

Reviewer meminta limitations dan further research suggestions ditambahkan ke conclusion.

### Status Perbaikan

Bagian ini perlu dimasukkan ke manuskrip. Dari sisi eksperimen, hasil revisi sudah menyediakan dasar untuk menulis limitations.

### Poin yang Perlu Ditambahkan ke Conclusion

Limitations:

- Data hanya mencakup tiga brand coffee shop.
- Data berasal dari Google Maps Reviews.
- Periode pengambilan data terbatas pada window penelitian.
- Sentimen data sangat imbalanced, sehingga perlu evaluasi yang hati-hati.
- Aspect extraction masih berbasis rule-based keyword lexicon.
- Decision Tree memiliki performa lebih rendah dibanding SVM dan IndoBERT.

Future research:

- Menambah jumlah brand dan outlet.
- Memperpanjang periode pengumpulan data.
- Menggunakan TripAdvisor, social media, atau food delivery platform.
- Mengembangkan ABSA berbasis transformer.
- Membandingkan IndoBERT, IndoBERTweet, XLM-R, atau multilingual BERT.
- Menguji pendekatan multi-label aspect classification.

## 5. Methodology Direkonstruksi

### Komentar Reviewer

Reviewer menyatakan methodology belum menjelaskan model building.

### Perbaikan yang Sudah Dilakukan

Sudah dibuat pipeline metodologi baru:

- Load dan standardisasi data dari 3 brand.
- Normalisasi label sentimen.
- Penambahan alias brand.
- Text preprocessing.
- PySastrawi stemming.
- TF-IDF vectorization.
- Rule-based aspect extraction.
- Train-test split 80:20.
- Oversampling hanya pada training data.
- Model training.
- Holdout evaluation.
- 5-fold stratified cross-validation.
- Per-class evaluation.
- Confusion matrix.
- Frequency analysis.
- IndoBERT fine-tuning.

### File Utama

- `revised_absa_pipeline.py`
- `revised_indobert_evaluation.py`

## 6. Aspect Extraction dan Sentiment Classification Dijelaskan

### Komentar Reviewer

Reviewer meminta penjelasan model atau algoritma yang digunakan untuk aspect extraction dan sentiment classification.

### Perbaikan yang Sudah Dilakukan

Aspect extraction sekarang eksplisit memakai rule-based keyword lexicon untuk lima aspek:

- ambiance
- packaging
- price
- service
- taste

Sentiment classification dilakukan dengan:

- Decision Tree
- Naive Bayes
- SVM
- Logistic Regression
- Random Forest
- Gradient Boosting
- IndoBERT

### Output ABSA

- `revised_outputs/revised_absa_aspect_level.csv`
- `revised_outputs/revised_aspect_distribution_table.csv`

## 7. Justifikasi Decision Tree dan State-of-the-Art

### Komentar Reviewer

Reviewer menyatakan Decision Tree kurang dijustifikasi dan tidak sejajar dengan state-of-the-art ABSA.

### Perbaikan yang Sudah Dilakukan

Eksperimen menunjukkan bahwa Decision Tree bukan model terbaik. Oleh karena itu, positioning paper harus direvisi:

- Decision Tree diposisikan sebagai interpretable baseline.
- SVM diposisikan sebagai classical ML terbaik.
- IndoBERT diposisikan sebagai transformer-based comparison.

### Hasil Decision Tree Revisi

| Metric | Score |
|---|---:|
| Accuracy | 0.7051 |
| Weighted Precision | 0.8672 |
| Weighted Recall | 0.7051 |
| Weighted F1 | 0.7508 |
| Macro F1 | 0.6131 |

### Catatan

Angka lama Decision Tree 88.96% sebaiknya tidak lagi digunakan sebagai klaim utama karena pipeline lama belum memakai full 3-brand revised setup dan berisiko data leakage dari oversampling sebelum split.

## 8. Kontribusi Paper Diperjelas

### Komentar Reviewer

Reviewer menyatakan kontribusi paper terbatas dan tidak menawarkan metode baru.

### Perbaikan yang Disarankan Berdasarkan Eksperimen

Kontribusi paper sebaiknya diposisikan sebagai applied comparative ABSA study, bukan novel ML method.

Kontribusi yang dapat ditulis:

- Studi komparatif ABSA pada Global, National, dan Local coffee brand di Indonesia.
- Menggunakan lima aspek customer experience: ambiance, packaging, price, service, taste.
- Menggabungkan rule-based ABSA dan supervised sentiment classification.
- Membandingkan Decision Tree, Naive Bayes, SVM, dan IndoBERT.
- Memberikan insight praktis untuk CRM dan e-business strategy.

## 9. Experimental Setup, Baseline Models, dan Evaluasi

### Komentar Reviewer

Reviewer menyatakan experimental setup kurang detail, tidak ada baseline comparison, dan evaluasi kurang mendalam.

### Perbaikan yang Sudah Dilakukan

Baseline model sudah ditambahkan dan dievaluasi:

- Decision Tree
- Naive Bayes
- SVM
- Logistic Regression
- Random Forest
- Gradient Boosting
- IndoBERT

Evaluasi yang sudah ditambahkan:

- Accuracy
- Precision weighted
- Recall weighted
- F1 weighted
- Precision macro
- Recall macro
- F1 macro
- Per-class precision, recall, F1
- Confusion matrix
- 5-fold stratified cross-validation

### Output File

- `revised_outputs/revised_model_holdout_metrics.csv`
- `revised_outputs/revised_model_cv_summary.csv`
- `revised_outputs/revised_classification_report.csv`
- `revised_outputs/revised_confusion_matrices.csv`
- `revised_outputs/revised_model_comparison_with_indobert.csv`

## 10. Hasil Model Final

### Holdout Model Comparison

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| IndoBERT | 0.9275 | 0.9275 | 0.8449 |
| SVM | 0.9242 | 0.9230 | 0.8327 |
| Random Forest | 0.9061 | 0.9077 | 0.8060 |
| Naive Bayes | 0.8929 | 0.8962 | 0.7850 |
| Logistic Regression | 0.8748 | 0.8842 | 0.7743 |
| Gradient Boosting | 0.8089 | 0.8320 | 0.7007 |
| Decision Tree | 0.7051 | 0.7508 | 0.6131 |

### 5-Fold Cross-Validation

CV dilakukan untuk model utama yang diminta reviewer:

| Model | Mean Accuracy | Mean Weighted F1 | Mean Macro F1 |
|---|---:|---:|---:|
| SVM | 0.9212 | 0.9182 | 0.8183 |
| Naive Bayes | 0.8945 | 0.8961 | 0.7806 |
| Decision Tree | 0.7073 | 0.7522 | 0.6089 |

## 11. Data Leakage dari Balancing Diperbaiki

### Masalah Sebelumnya

Balanced dataset dibuat sebelum train-test split. Hal ini berisiko membuat duplikasi review masuk ke train dan test sekaligus.

### Perbaikan yang Sudah Dilakukan

Pipeline revisi sekarang melakukan:

1. Train-test split terlebih dahulu.
2. Oversampling hanya pada training data.
3. Test data dibiarkan natural/tidak di-oversampling.

### Dampak

Hasil model menjadi lebih realistis. Decision Tree turun dari angka lama 88.96% menjadi 70.51% accuracy pada holdout revised setup.

## 12. Bug Dashboard Streamlit Diperbaiki

### Masalah

Model Starbucks di dashboard sebelumnya dilatih menggunakan dataset Kopi Kenangan.

### Perbaikan

Di `streamlit_app.py`, baris training Starbucks sudah diperbaiki dari:

```python
'Starbucks': (kenangan, GradientBoostingClassifier(...))
```

menjadi:

```python
'Starbucks': (starbucks, GradientBoostingClassifier(...))
```

## 13. IndoBERT Sudah Dilakukan

### Alasan

Reviewer menyarankan agar pendekatan Decision Tree dibandingkan dengan state-of-the-art ABSA atau model transformer.

### Perbaikan

Eksperimen IndoBERT sudah dijalankan menggunakan:

```bash
python revised_indobert_evaluation.py --epochs 3 --batch-size 16 --eval-batch-size 32 --max-length 128 --no-save-model --log-every 25
```

Model:

```text
indobenchmark/indobert-base-p1
```

Hardware:

```text
NVIDIA GeForce RTX 3070 Ti Laptop GPU
```

### Hasil IndoBERT

| Metric | Score |
|---|---:|
| Accuracy | 0.9275 |
| Weighted Precision | 0.9275 |
| Weighted Recall | 0.9275 |
| Weighted F1 | 0.9275 |
| Macro F1 | 0.8449 |

### Output File

- `revised_outputs/revised_indobert_holdout_metrics.csv`
- `revised_outputs/revised_indobert_classification_report.csv`
- `revised_outputs/revised_indobert_confusion_matrix.csv`
- `revised_outputs/revised_indobert_training_history.csv`
- `revised_outputs/revised_model_comparison_with_indobert.csv`

## 14. E-Business dan CRM Section

### Komentar Reviewer

Reviewer menilai pembahasan e-business dan CRM terlalu panjang dan kurang langsung terkait kontribusi teknis.

### Status Perbaikan

Ini adalah revisi naskah. Belum bisa diterapkan langsung karena file sumber paper belum tersedia.

### Rekomendasi

Bagian e-business dan CRM sebaiknya dipadatkan dan hanya dikaitkan dengan:

- digital customer experience
- online review analytics
- CRM insight berbasis sentiment
- keputusan bisnis coffee brand

Tambahkan subbagian ABSA literature yang membahas:

- rule-based ABSA
- machine learning ABSA
- transformer-based ABSA
- posisi IndoBERT sebagai pembanding terbaru

## 15. File yang Harus Dipakai untuk Revisi Manuskrip

Gunakan file berikut sebagai sumber tabel dan angka final:

- `revised_outputs/REVISED_ANALYSIS_SUMMARY.md`
- `revised_outputs/revised_clean_reviews.csv`
- `revised_outputs/revised_absa_aspect_level.csv`
- `revised_outputs/revised_aspect_distribution_table.csv`
- `revised_outputs/revised_model_holdout_metrics.csv`
- `revised_outputs/revised_model_cv_summary.csv`
- `revised_outputs/revised_classification_report.csv`
- `revised_outputs/revised_confusion_matrices.csv`
- `revised_outputs/revised_sentiment_word_frequency.csv`
- `revised_outputs/revised_indobert_holdout_metrics.csv`
- `revised_outputs/revised_indobert_classification_report.csv`
- `revised_outputs/revised_indobert_confusion_matrix.csv`
- `revised_outputs/revised_model_comparison_with_indobert.csv`

## 16. Bagian yang Masih Perlu Diedit di Manuskrip

Coding dan eksperimen sudah selesai, tetapi PDF belum bisa diedit langsung karena hanya file PDF yang tersedia. Untuk menerapkan revisi final, perlu file sumber seperti `.docx` atau `.tex`.

Bagian manuskrip yang perlu diedit:

- Abstract
- Introduction
- Literature Review
- Methodology
- Results
- Discussion
- Conclusion
- Tables 1-7
- Response letter

## Kesimpulan

Secara eksperimen, revisi utama sudah selesai. Project sekarang memiliki pipeline yang lebih kuat, transparan, dan sesuai komentar reviewer:

- Dataset 3 brand sudah distandardisasi.
- ABSA pipeline sudah eksplisit.
- Baseline comparison sudah tersedia.
- Evaluation sudah lebih lengkap.
- Data leakage dari oversampling sudah diperbaiki.
- Table 7 sudah dibersihkan dari kata netral seperti coffee/kopi.
- IndoBERT sudah dijalankan sebagai transformer comparison.
- Hasil akhir menunjukkan IndoBERT sebagai model terbaik, diikuti SVM sebagai classical ML terbaik.

Langkah berikutnya adalah menerapkan angka dan narasi baru ini ke file manuskrip sumber.
