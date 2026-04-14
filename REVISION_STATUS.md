# Reviewer Revision Status

This project now has a reviewer-ready analysis path.

## Completed

- Added `revised_absa_pipeline.py` as the revised source of truth.
- Standardized all three brand datasets into 3,034 review rows.
- Added brand aliases: Global Brand, National Brand, and Local Brand.
- Added explicit rule-based aspect extraction for ambiance, packaging, price, service, and taste.
- Added PySastrawi stemming support.
- Fixed the evaluation leakage risk by applying oversampling only after the train split, and only to training data.
- Added reviewer-requested baselines: Naive Bayes and SVM.
- Added 5-fold stratified cross-validation for Decision Tree, Naive Bayes, and SVM.
- Added per-class precision, recall, F1, macro averages, weighted averages, and confusion matrices.
- Regenerated sentiment-bearing word frequency tables with neutral domain nouns such as coffee/kopi removed.
- Fixed the Streamlit bug where the Starbucks model was trained on the Kopi Kenangan dataset.
- Added and ran `revised_indobert_evaluation.py` using `indobenchmark/indobert-base-p1` on GPU.
- Added a combined classical ML plus IndoBERT comparison table.

## Revised Key Results

- Final data: 3,034 reviews.
- Holdout best model: SVM, weighted F1 = 0.9230.
- 5-fold CV best model: SVM, mean weighted F1 = 0.9182.
- IndoBERT holdout result: weighted F1 = 0.9275, macro F1 = 0.8449.
- Revised Decision Tree holdout metrics for Table 1:
  - Accuracy = 0.7051.
  - Weighted precision = 0.8672.
  - Weighted recall = 0.7051.
  - Weighted F1 = 0.7508.
  - Macro F1 = 0.6131.

## Files To Use

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

## IndoBERT Run Command

```bash
python revised_indobert_evaluation.py --epochs 3 --batch-size 16 --eval-batch-size 32 --max-length 128 --no-save-model --log-every 25
```

The completed run used CUDA on an NVIDIA GeForce RTX 3070 Ti Laptop GPU.

## Manuscript Note

The PDF itself was not edited because only the compiled PDF is present in this
folder. The manuscript source file, such as Word or LaTeX, is needed to apply
the textual revisions directly.
