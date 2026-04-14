# Revised Analysis Summary

## Data

- Final standardized review rows: 3,034
- Lexicon-generated aspect rows: 3,672
- PySastrawi stemming used: yes

| brand         | brand_alias    |   negative |   positive |   total |
|:--------------|:---------------|-----------:|-----------:|--------:|
| Kopi Kenangan | National Brand |        192 |       1259 |    1451 |
| Kopi Nako     | Local Brand    |        116 |        884 |    1000 |
| Starbucks     | Global Brand   |        101 |        482 |     583 |

## Holdout Model Results

| Model               |   Accuracy |   Precision_Weighted |   Recall_Weighted |   F1_Weighted |   F1_Macro |
|:--------------------|-----------:|---------------------:|------------------:|--------------:|-----------:|
| SVM                 |     0.9242 |               0.9221 |            0.9242 |        0.923  |     0.8327 |
| Random Forest       |     0.9061 |               0.9097 |            0.9061 |        0.9077 |     0.806  |
| Naive Bayes         |     0.8929 |               0.9006 |            0.8929 |        0.8962 |     0.785  |
| Logistic Regression |     0.8748 |               0.9021 |            0.8748 |        0.8842 |     0.7743 |
| Gradient Boosting   |     0.8089 |               0.8813 |            0.8089 |        0.832  |     0.7007 |
| Decision Tree       |     0.7051 |               0.8672 |            0.7051 |        0.7508 |     0.6131 |

## IndoBERT Holdout Result

| Model | Accuracy | Precision_Weighted | Recall_Weighted | F1_Weighted | F1_Macro |
|:------|---------:|-------------------:|----------------:|------------:|---------:|
| indobenchmark/indobert-base-p1 | 0.9275 | 0.9275 | 0.9275 | 0.9275 | 0.8449 |

See `revised_model_comparison_with_indobert.csv` for the combined ranking.

## Five-Fold Cross-Validation

CV is reported for the reviewer-requested core models: Decision Tree, Naive Bayes, and SVM.

| Model         |   Accuracy_Mean |   Accuracy_Std |   F1_Weighted_Mean |   F1_Weighted_Std |   F1_Macro_Mean |   F1_Macro_Std |   Precision_Weighted_Mean |   Recall_Weighted_Mean |
|:--------------|----------------:|---------------:|-------------------:|------------------:|----------------:|---------------:|--------------------------:|-----------------------:|
| SVM           |          0.9212 |         0.0169 |             0.9182 |            0.0191 |          0.8183 |         0.0459 |                    0.917  |                 0.9212 |
| Naive Bayes   |          0.8945 |         0.0142 |             0.8961 |            0.0151 |          0.7806 |         0.0359 |                    0.8988 |                 0.8945 |
| Decision Tree |          0.7073 |         0.0171 |             0.7522 |            0.0141 |          0.6089 |         0.0215 |                    0.8599 |                 0.7073 |

## Decision Tree Metrics for Revised Table 1

- Accuracy: 0.7051
- Weighted precision: 0.8672
- Weighted recall: 0.7051
- Weighted F1-score: 0.7508
- Macro F1-score: 0.6131

## Recommended Manuscript Notes

- Best holdout model by weighted F1: SVM (0.9230).
- Best 5-fold CV model by mean weighted F1: SVM (0.9182).
- Describe Decision Tree as an interpretable baseline, not as state of the art.
- Oversampling is applied only to training partitions to prevent duplicate-review leakage into the test set.
- Use revised_sentiment_word_frequency.csv for Tables 5-7; neutral domain nouns such as coffee/kopi are filtered.

## Files To Use

- revised_clean_reviews.csv
- revised_absa_aspect_level.csv
- revised_aspect_distribution_table.csv
- revised_model_holdout_metrics.csv
- revised_model_cv_summary.csv
- revised_classification_report.csv
- revised_confusion_matrices.csv
- revised_sentiment_word_frequency.csv
- revised_indobert_holdout_metrics.csv
- revised_indobert_classification_report.csv
- revised_indobert_confusion_matrix.csv
- revised_model_comparison_with_indobert.csv

## Top Sentiment Words Preview

| brand         | brand_alias    | sentiment   |   rank | word          |   count |
|:--------------|:---------------|:------------|-------:|:--------------|--------:|
| Kopi Kenangan | National Brand | negative    |      1 | slow          |      57 |
| Kopi Kenangan | National Brand | negative    |      2 | bad           |      18 |
| Kopi Kenangan | National Brand | negative    |      3 | less          |      18 |
| Kopi Kenangan | National Brand | negative    |      4 | inconsistent  |       9 |
| Kopi Kenangan | National Brand | negative    |      5 | not friendly  |       9 |
| Kopi Kenangan | National Brand | negative    |      6 | disappointed  |       8 |
| Kopi Kenangan | National Brand | negative    |      7 | not delicious |       7 |
| Kopi Kenangan | National Brand | negative    |      8 | expensive     |       3 |
| Kopi Kenangan | National Brand | positive    |      1 | delicious     |     326 |
| Kopi Kenangan | National Brand | positive    |      2 | cozy          |     305 |
| Kopi Kenangan | National Brand | positive    |      3 | good          |     255 |
| Kopi Kenangan | National Brand | positive    |      4 | friendly      |     246 |
| Kopi Kenangan | National Brand | positive    |      5 | clean         |     117 |
| Kopi Kenangan | National Brand | positive    |      6 | favorite      |      82 |
| Kopi Kenangan | National Brand | positive    |      7 | spacious      |      82 |
| Kopi Kenangan | National Brand | positive    |      8 | fast          |      50 |
| Kopi Nako     | Local Brand    | negative    |      1 | less          |      21 |
| Kopi Nako     | Local Brand    | negative    |      2 | expensive     |      14 |
| Kopi Nako     | Local Brand    | negative    |      3 | slow          |      12 |
| Kopi Nako     | Local Brand    | negative    |      4 | bad           |       4 |
| Kopi Nako     | Local Brand    | negative    |      5 | inconsistent  |       4 |
| Kopi Nako     | Local Brand    | negative    |      6 | not delicious |       4 |
| Kopi Nako     | Local Brand    | negative    |      7 | not friendly  |       4 |
| Kopi Nako     | Local Brand    | negative    |      8 | disappointed  |       3 |
