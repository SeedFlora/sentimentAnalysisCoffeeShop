# ☕ Sentiment Analysis: Coffee Shop Reviews
## Indonesian Coffee Chain Reviews Classification using Machine Learning

> Reviewer revision note: use `revised_absa_pipeline.py` and the files in
> `revised_outputs/` for the revised manuscript. The older metrics below are
> retained as historical exploratory output and are superseded by the revised
> 3-brand pipeline with training-only oversampling and 5-fold CV.

<div align="center">

![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11+-green.svg?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg?style=flat-square)
![Models](https://img.shields.io/badge/Models-ML%20%2B%20IndoBERT-orange.svg?style=flat-square)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-92.75%25-success.svg?style=flat-square)
![Datasets](https://img.shields.io/badge/Brands%20Analyzed-3-blue.svg?style=flat-square)

**Advanced NLP & Machine Learning | Streamlit Dashboard | Production-Ready**

[🔗 Live Dashboard](#-deployment) • [📊 Documentation](#-table-of-contents) • [🚀 Quick Start](#-quick-start) • [📈 Results](#-model-performance)

</div>

---

## 📋 Table of Contents

| Section | Link |
|---------|------|
| 🎯 Overview | [View](#-overview) |
| ✅ Latest Revision Results | [View](#-latest-reviewer-ready-results) |
| 📊 Dataset | [View](#-dataset-description) |
| 📁 Structure | [View](#-project-structure) |
| 🚀 Quick Start | [View](#-quick-start) |
| ⚙️ Installation | [View](#-installation--setup) |
| 💻 Usage | [View](#-usage) |
| 📈 Results | [View](#-model-performance) |
| ☕ Per-Brand Analysis | [View](#-per-brand-analysis--metrics) |
| 🔍 Key Findings | [View](#-key-findings) |
| 📊 Metrics Explained | [View](#-evaluation-metrics-explained) |
| 📁 Output Files | [View](#-output-files) |
| 🔧 Deployment | [View](#-deployment) |
| 📋 Requirements | [View](#-requirements) |

---

## 🎯 Overview

This project implements a **comprehensive sentiment analysis system** for Indonesian coffee shop reviews from **3 major brands**:
- ☕ **Kopi Nako** (1,000 reviews)
- ☕ **Starbucks** (583 reviews)  
- ☕ **Kopi Kenangan** (1,451 reviews)

The revised reviewer-ready experiment uses **classical machine learning baselines plus IndoBERT** to classify customer reviews as **positive** or **negative** sentiment. The best revised holdout result is **92.75% accuracy / 0.9275 weighted F1** using **IndoBERT**.

### 🎓 What You'll Learn
- ✨ Text preprocessing & NLP for Indonesian language
- 🤖 Train & evaluate classical ML models and IndoBERT
- 📊 Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix)
- 🎨 Professional data visualization with Matplotlib & Seaborn
- 🚀 Deploy interactive dashboard with Streamlit
- 📈 Per-brand sentiment analysis & model comparison  

---

## ✅ Latest Reviewer-Ready Results

Use these revised files for the manuscript response:

- `revised_absa_pipeline.py`
- `revised_indobert_evaluation.py`
- `revised_outputs/REVISED_ANALYSIS_SUMMARY.md`
- `revised_outputs/revised_model_comparison_with_indobert.csv`
- `REVISI_REVIEWER_YANG_SUDAH_DILAKUKAN.md`
- `REVISION_STATUS.md`

### Final Dataset

| Brand Alias | Brand Source | Reviews |
|---|---|---:|
| Global Brand | Starbucks | 583 |
| National Brand | Kopi Kenangan | 1,451 |
| Local Brand | Kopi Nako | 1,000 |
| **Total** | **3 brands** | **3,034** |

### Revised Holdout Model Comparison

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| **IndoBERT** (`indobenchmark/indobert-base-p1`) | **0.9275** | **0.9275** | **0.8449** |
| SVM | 0.9242 | 0.9230 | 0.8327 |
| Random Forest | 0.9061 | 0.9077 | 0.8060 |
| Naive Bayes | 0.8929 | 0.8962 | 0.7850 |
| Logistic Regression | 0.8748 | 0.8842 | 0.7743 |
| Gradient Boosting | 0.8089 | 0.8320 | 0.7007 |
| Decision Tree | 0.7051 | 0.7508 | 0.6131 |

### Revised 5-Fold Cross-Validation

CV was run for the reviewer-requested core models:

| Model | Mean Accuracy | Mean Weighted F1 | Mean Macro F1 |
|---|---:|---:|---:|
| SVM | 0.9212 | 0.9182 | 0.8183 |
| Naive Bayes | 0.8945 | 0.8961 | 0.7806 |
| Decision Tree | 0.7073 | 0.7522 | 0.6089 |

### IndoBERT Run

The completed IndoBERT experiment used CUDA on an NVIDIA GeForce RTX 3070 Ti Laptop GPU.

```bash
python revised_indobert_evaluation.py --epochs 3 --batch-size 16 --eval-batch-size 32 --max-length 128 --no-save-model --log-every 25
```

### Important Revision Notes

- The older **Decision Tree 88.96%** result is retained only as historical exploratory output.
- The revised pipeline fixes the previous leakage risk by applying oversampling only to the training split.
- Decision Tree is now positioned as an interpretable baseline.
- SVM is the strongest classical ML model.
- IndoBERT is the strongest overall model in the revised holdout comparison.
- Neutral domain terms such as `coffee` and `kopi` were removed from the revised frequency table.

---

## � Quick Start

### Option 1: Run Revised Reviewer Pipeline
```bash
cd d:\skripsi angel
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python revised_absa_pipeline.py
```

### Option 2: Run IndoBERT on GPU/CPU
```bash
python revised_indobert_evaluation.py --epochs 3 --batch-size 16 --eval-batch-size 32 --max-length 128 --no-save-model --log-every 25
```

### Option 3: Run Historical Local Analysis
```bash
# 1. Clone repository
cd d:\skripsi angel

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run analysis
python sentiment_analysis.py

# 5. View results (PNG files generated automatically)
```

### Option 4: Deploy to Streamlit Cloud (3 minutes)
```bash
# Go to https://streamlit.io/cloud
# Sign in with GitHub
# Deploy: SeedFlora/sentimentAnalysisCoffeeShop → main → streamlit_app.py
# Access: https://sentimentanalysiscoffeeshop.streamlit.app
```

---

## 📊 Dataset Description

### 📈 Dataset Overview
| Brand | Reviews | Class Balance | Language |
|-------|---------|---|----------|
| **Kopi Nako** | 1,000 | 88% Pos / 12% Neg | Indonesian |
| **Starbucks** | 583 | 82% Pos / 18% Neg | Indonesian |
| **Kopi Kenangan** | 1,451 | 85% Pos / 15% Neg | Indonesian |
| **TOTAL COMBINED** | **3,034** | **86% Pos / 14% Neg** | **Indonesian** |

### Available Columns:
- **title**: Review title
- **text**: Full review text (PRIMARY FEATURE)
- **sentiment**: Ground truth label (positive/negative)
- **stars**: Review rating (1-5)
- **taste**, **service**, **ambience**, **price**, **packaging**: Feature ratings

---

## 📁 Project Structure

```
d:\skripsi angel/
├── venv/                                  # Virtual environment
├── kopinako_main_analysis.csv            # Kopi Nako reviews dataset
├── starbucks_detailed_reviews.csv        # Starbucks reviews dataset
├── Kopi_Kenangan.xlsx                    # Kopi Kenangan reviews dataset
│
├── PYTHON SCRIPTS:
├── sentiment_analysis.py                 # Combined analysis (Kopi Nako + Starbucks)
├── per_brand_analysis.py                 # Per-brand balancing analysis
├── per_brand_model_evaluation.py         # Per-brand model evaluation
├── explore_csvs.py                       # Data exploration script
│
├── OUTPUT FILES - COMBINED ANALYSIS:
├── model_performance_results.csv         # Model performance comparison
├── model_comparison.png                  # Performance visualization
├── roc_curve.png                        # ROC curve for best model
├── sentiment_distribution.png            # Sentiment distribution charts
├── feature_importance_rf.png             # Random Forest top features
├── feature_importance_lr.png             # Logistic Regression coefficients
│
├── OUTPUT FILES - PER-BRAND ANALYSIS:
├── per_brand_evaluation_results.csv      # Per-brand metrics (Accuracy, Precision, Recall, F1, AUC)
├── per_brand_metrics_comparison.png      # Per-brand metrics visualization
├── per_brand_models_comparison.png       # All 6 models comparison per brand
├── per_brand_metrics_heatmap.png         # Metrics heatmap visualization
├── per_brand_f1_radar.png               # F1-Score radar chart
├── brand_sentiment_before_after.png      # Balance before/after comparison
├── brand_sentiment_pie_charts.png        # Sentiment distribution pie charts
├── brand_balance_ratio.png              # Balance ratio improvement chart
│
├── BALANCED DATASETS:
├── kopi_nako_balanced.csv               # Balanced Kopi Nako (1,768 samples)
├── starbucks_balanced.csv               # Balanced Starbucks (964 samples)
├── kopi_kenangan_balanced.csv           # Balanced Kopi Kenangan (2,518 samples)
│
├── STREAMLIT APP:
├── streamlit_app.py                     # Interactive dashboard
├── requirements.txt                      # Python dependencies
│
└── README.md                             # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Windows, Linux, or macOS
- 4GB RAM minimum

### Step 1: Clone or Download Repository
```bash
cd d:\skripsi angel
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment
```bash
# Windows
.\venv\Scripts\Activate.ps1

# Linux/macOS
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Run Revised Reviewer Analysis
```bash
python revised_absa_pipeline.py
python revised_indobert_evaluation.py --epochs 3 --batch-size 16 --eval-batch-size 32 --max-length 128 --no-save-model --log-every 25
```

### Run Historical Analysis
```bash
python sentiment_analysis.py
```

### Revised Expected Output
The revised scripts will:
1. Standardize all 3 brand datasets into 3,034 reviews
2. Run explicit rule-based ABSA aspect extraction
3. Train and evaluate classical ML baselines
4. Run 5-fold stratified cross-validation for core baselines
5. Fine-tune IndoBERT for sentiment classification
6. Generate revised tables, metrics, confusion matrices, and manuscript notes under `revised_outputs/`

### Historical Expected Output
The historical script will:
1. Load both CSV datasets
2. Clean and preprocess text data
3. Extract TF-IDF features
4. Train 6 machine learning models
5. Evaluate each model with comprehensive metrics
6. Generate visualizations and reports
7. Save results to CSV and PNG files

### Runtime
- Estimated execution time: **2-3 minutes** (depending on machine specs)
- Generates 6 output files

---

## 📈 Historical Model Performance

> The section below is retained for traceability of the original exploratory run.
> For the revised manuscript and reviewer response, use
> [Latest Reviewer-Ready Results](#-latest-reviewer-ready-results).

### 🏆 Overall Rankings (Combined Dataset: 1,583 reviews)

<div align="center">

![Model Comparison](https://raw.githubusercontent.com/SeedFlora/sentimentAnalysisCoffeeShop/main/model_comparison.png)

</div>

<div align="center">

| Rank | 🥇 Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|:----:|:---------|:--------:|:---------:|:------:|:--------:|:-------:|
| 🥇 | **Decision Tree** | **88.96%** | **87.56%** | **88.96%** | **86.94%** | 70.84% |
| 🥈 | Gradient Boosting | 88.64% | 87.22% | 88.64% | 86.18% | **81.67%** |
| 🥉 | Support Vector Machine | 87.70% | 86.26% | 87.70% | 83.84% | **90.46%** ⭐ |
| 4️⃣ | Multinomial Naive Bayes | 87.70% | 87.18% | 87.70% | 83.44% | 83.88% |
| 5️⃣ | Logistic Regression | 86.75% | 83.49% | 86.75% | 81.94% | 88.65% |
| 6️⃣ | Random Forest | 86.44% | 74.71% | 86.44% | 80.15% | 85.67% |

</div>

### 📊 Best Models Explained

**🏆 Best Overall: Decision Tree (88.96% Accuracy)**
```
✓ Highest accuracy and F1-score
✓ Balanced performance across metrics
✓ Fast inference time
✓ Interpretable results
```

**⭐ Best AUC-ROC: Support Vector Machine (90.46%)**
```
✓ Excellent class discrimination
✓ Most reliable probability estimates
✓ Best for ranking predictions
✓ Superior generalization
```

---

## ☕ Per-Brand Analysis & Metrics

### 🎯 Per-Brand Model Performance

<div align="center">

![Per-Brand Metrics Comparison](https://raw.githubusercontent.com/SeedFlora/sentimentAnalysisCoffeeShop/main/per_brand_metrics_comparison.png)

</div>

<details>
<summary><b>KOPI NAKO</b> (1,768 balanced samples) - Click to expand</summary>

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| 🥇 **SVM** | **97.74%** | **97.84%** | **97.74%** | **97.74%** | **99.99%** |
| Random Forest | 97.46% | 97.58% | 97.46% | 97.46% | 99.98% |
| Logistic Regression | 95.48% | 95.53% | 95.48% | 95.48% | 99.95% |
| Gradient Boosting | 92.94% | 93.81% | 92.94% | 92.90% | 99.77% |
| Decision Tree | 92.66% | 93.60% | 92.66% | 92.62% | 99.65% |
| Naive Bayes | 91.53% | 92.30% | 91.53% | 91.49% | 99.45% |

**✨ Best Model: SVM (97.74% F1-Score)**

</details>

<details>
<summary><b>STARBUCKS</b> (964 balanced samples) - Click to expand</summary>

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| 🥇 **Gradient Boosting** | **95.34%** | **95.38%** | **95.34%** | **95.34%** | **99.88%** |
| Logistic Regression | 91.71% | 91.71% | 91.71% | 91.71% | 99.65% |
| SVM | 91.71% | 91.71% | 91.71% | 91.71% | 99.62% |
| Random Forest | 90.67% | 90.96% | 90.67% | 90.66% | 99.58% |
| Naive Bayes | 87.56% | 87.98% | 87.56% | 87.53% | 99.12% |
| Decision Tree | 78.24% | 84.81% | 78.24% | 77.13% | 98.15% |

**✨ Best Model: Gradient Boosting (95.34% F1-Score)**

</details>

<details>
<summary><b>KOPI KENANGAN</b> (2,518 balanced samples) - Click to expand</summary>

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| 🥇 **SVM** | **96.43%** | **96.44%** | **96.43%** | **96.43%** | **99.99%** |
| Logistic Regression | 95.24% | 95.31% | 95.24% | 95.24% | 99.96% |
| Gradient Boosting | 94.84% | 94.84% | 94.84% | 94.84% | 99.91% |
| Random Forest | 94.44% | 94.58% | 94.44% | 94.44% | 99.89% |
| Decision Tree | 92.66% | 92.90% | 92.66% | 92.65% | 99.69% |
| Naive Bayes | 89.88% | 90.06% | 89.88% | 89.87% | 99.24% |

**✨ Best Model: SVM (96.43% F1-Score)**

</details>

### 🎖️ Brand Rankings

<div align="center">

![Per-Brand F1 Radar](https://raw.githubusercontent.com/SeedFlora/sentimentAnalysisCoffeeShop/main/per_brand_f1_radar.png)

</div>

<div align="center">

| Rank | Brand | Best F1-Score | Best Model | Status |
|:----:|:------|:-------------:|:-----------|:-------|
| 🥇 | **Kopi Nako** | **97.74%** | SVM | 🟢 EXCELLENT |
| 🥈 | **Kopi Kenangan** | **96.43%** | SVM | 🟢 EXCELLENT |
| 🥉 | **Starbucks** | **95.34%** | Gradient Boosting | 🟢 VERY GOOD |

</div>

### 📊 Per-Brand Visualization Files

**📈 Metrics Visualizations:**
- `per_brand_metrics_comparison.png` - 📊 Accuracy, Precision, Recall, F1 comparison
- `per_brand_metrics_heatmap.png` - 🔥 Heatmap of all metrics per brand
- `per_brand_models_comparison.png` - 📊 All 6 models performance per brand
- `per_brand_f1_radar.png` - 🎯 Radar chart of F1-Scores across brands

**⚖️ Balance Visualizations:**
- `brand_sentiment_before_balancing.png` - 📊 Original imbalanced distribution
- `brand_sentiment_before_after.png` - 🔄 Before/after balancing comparison
- `brand_sentiment_pie_charts.png` - 🥧 Pie charts of sentiment per brand
- `brand_balance_ratio.png` - 📈 Balance improvement metrics

<div align="center">

![Brand Balance Ratio](https://raw.githubusercontent.com/SeedFlora/sentimentAnalysisCoffeeShop/main/brand_balance_ratio.png)

![Sentiment Before After](https://raw.githubusercontent.com/SeedFlora/sentimentAnalysisCoffeeShop/main/brand_sentiment_before_after.png)

</div>

---

## 🔍 Key Findings

### 1️⃣ Confusion Matrix Analysis (Best Model: Decision Tree)
```
                Predicted Negative    Predicted Positive
Actual Negative:    13 (TN)              30 (FP)
Actual Positive:     5 (FN)             269 (TP)
```

**Interpretation:**
- True Positive Rate (TPR/Recall): 98.2% - Excellent at detecting positive reviews
- True Negative Rate (TNR): 30.2% - Moderate at detecting negative reviews
- False Positive Rate: 69.8% - Sometimes misclassifies negative as positive

### 2️⃣ Sentiment Distribution

<div align="center">

![Sentiment Distribution](https://raw.githubusercontent.com/SeedFlora/sentimentAnalysisCoffeeShop/main/sentiment_distribution.png)

</div>

- **High Imbalance**: 86.3% positive vs 13.7% negative
- **Implication**: Models are biased toward positive sentiment
- **Recommendation**: Consider balanced sampling or class weights for production

### 3️⃣ Top Features for Sentiment Classification

<div align="center">

![Feature Importance - Random Forest](https://raw.githubusercontent.com/SeedFlora/sentimentAnalysisCoffeeShop/main/feature_importance_rf.png)

![Feature Importance - Logistic Regression](https://raw.githubusercontent.com/SeedFlora/sentimentAnalysisCoffeeShop/main/feature_importance_lr.png)

</div>

<div align="center">

![Sentiment Distribution](https://raw.githubusercontent.com/SeedFlora/sentimentAnalysisCoffeeShop/main/sentiment_distribution.png)

</div>

**Words Most Associated with POSITIVE Sentiment:**
- 🟢 **enak** (delicious)
- 🟢 **juara** (champion/excellent)
- 🟢 **baik** (good)
- 🟢 **nyaman** (comfortable)
- 🟢 **ramah** (friendly)

**Words Most Associated with NEGATIVE Sentiment:**
- 🔴 **ramai** (crowded)
- 🔴 **lama** (slow/long)
- 🔴 **bau** (smell)
- 🔴 **pengap** (stuffy)
- 🔴 **mahal** (expensive)

### 4️⃣ Text Preprocessing Impact
- ✨ Removed special characters and numbers
- ✨ Converted to lowercase
- ✨ Applied Indonesian stopword removal
- ✨ Extracted 1,800 TF-IDF features

---

## 📊 Evaluation Metrics Explained

<div align="center">

| Metric | Formula | Best Range | What It Measures |
|--------|---------|-----------|------------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 0-1 | Overall correctness |
| **Precision** | TP/(TP+FP) | 0-1 | Positive prediction accuracy |
| **Recall** | TP/(TP+FN) | 0-1 | Positive identification rate |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | 0-1 | Balance of precision & recall |
| **AUC-ROC** | Area under ROC curve | 0-1 | Class discrimination ability |

</div>

### 📖 Detailed Explanations

#### 🎯 Accuracy
```
What: Percentage of correct predictions overall
Formula: (TP + TN) / (TP + TN + FP + FN)
Best Value: 100% (unrealistic)
Our Best: 88.96% (Decision Tree)
Meaning: Out of 100 predictions, ~89 are correct
Use: General performance overview
⚠️ Note: Not ideal for imbalanced datasets
```

#### 🎯 Precision  
```
What: Accuracy of positive predictions
Formula: TP / (TP + FP)
Best Value: 100%
Our Best: 87.56% (Decision Tree)
Meaning: When model says "positive", it's right 87.56% of the time
Use: When false positives are costly
```

#### 🎯 Recall (Sensitivity)
```
What: Percentage of actual positives identified
Formula: TP / (TP + FN)
Best Value: 100%
Our Best: 88.96% (Decision Tree)
Meaning: Model catches 88.96% of actual positive reviews
Use: When missing positives is costly
```

#### 🎯 F1-Score
```
What: Harmonic mean of Precision and Recall
Formula: 2 × (Precision × Recall) / (Precision + Recall)
Best Value: 1.0
Our Best: 0.8694 (Decision Tree)
Range: 0 to 1
Use: BEST for imbalanced datasets (our case!)
```

#### 🎯 AUC-ROC
```
What: Model's ability to distinguish between classes
Range: 0.5 (random) to 1.0 (perfect)
Our Best: 0.9046 (Support Vector Machine)
Meaning: 90.46% probability of correctly ranking
         a positive review higher than negative
Use: Best for probability-based predictions
```

<div align="center">

![ROC Curve](https://raw.githubusercontent.com/SeedFlora/sentimentAnalysisCoffeeShop/main/roc_curve.png)

</div>

### 🔢 Confusion Matrix Explained

```
                  Predicted Negative    Predicted Positive
Actual Negative:       TN                      FP
Actual Positive:       FN                      TP

TP (True Positive):      Correctly predicted positive ✓
TN (True Negative):      Correctly predicted negative ✓
FP (False Positive):     Incorrectly predicted positive ✗ (Type I Error)
FN (False Negative):     Incorrectly predicted negative ✗ (Type II Error)
```

---

## 📁 Output Files

### 1. **model_performance_results.csv**
Comparison of all models across metrics:
```csv
Model,Accuracy,Precision,Recall,F1-Score,AUC-ROC
Decision Tree,0.8896,0.8756,0.8896,0.8694,0.7084
...
```

### 2. **model_comparison.png** 
3-panel visualization:
- Accuracy comparison bar chart
- All metrics comparison grouped bar chart
- Top 2 confusion matrices heatmaps

### 3. **roc_curve.png**
ROC curve for best model showing:
- True Positive Rate vs False Positive Rate
- AUC score
- Diagonal reference line (random classifier)

### 4. **sentiment_distribution.png**
2-panel distribution charts:
- Overall sentiment distribution
- Train-test split distribution

### 5. **feature_importance_rf.png**
Top 15 most important features from Random Forest model with relative importance values

### 6. **feature_importance_lr.png**
Top 15 feature coefficients from Logistic Regression:
- Red bars: Negative sentiment indicators
- Blue bars: Positive sentiment indicators

---

## 📋 Requirements

### Python Packages
```
pandas==2.3.3
numpy==2.4.0
matplotlib==3.10.8
seaborn==0.13.2
scikit-learn==1.8.0
nltk==3.9.2
openpyxl==3.1.5
```

### System Requirements
- OS: Windows 7+, Linux, or macOS
- Python: 3.11 or higher
- RAM: 4GB minimum (8GB recommended)
- Disk Space: 500MB (including virtual environment)

---

## 🔧 Technical Implementation

### Data Preprocessing Pipeline
1. **Loading**: Read CSV files using pandas
2. **Cleaning**: Remove NaN values, filter empty texts
3. **Text Normalization**: Lowercase conversion, special character removal
4. **Feature Extraction**: TF-IDF vectorization with:
   - Max features: 1,800
   - N-grams: (1, 2)
   - Min document frequency: 2
   - Max document frequency: 80%
5. **Train-Test Split**: 80-20 stratified split

### Machine Learning Models

| Model | Type | Key Parameters |
|-------|------|-----------------|
| Decision Tree | Tree-based | max_depth=15 |
| Gradient Boosting | Ensemble | n_estimators=80, max_depth=5 |
| Support Vector Machine | Kernel-based | kernel='rbf', probability=True |
| Multinomial Naive Bayes | Probabilistic | Default |
| Logistic Regression | Linear | max_iter=1000, solver='lbfgs' |
| Random Forest | Ensemble | n_estimators=80, max_depth=15 |

---

## 💡 Recommendations

### For Production Deployment:
1. **Use Support Vector Machine** for best generalization (highest AUC-ROC: 0.9046)
2. **Implement ensemble voting** combining top 3 models for robustness
3. **Handle class imbalance** using:
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Class weights in model training
   - Stratified cross-validation
4. **Monitor model drift** - retrain monthly with new reviews

### For Model Improvement:
1. **Collect more negative examples** to balance dataset
2. **Implement deep learning** (BERT, LSTM) for better contextual understanding
3. **Add n-gram analysis** and sentiment lexicon features
4. **Apply hyperparameter tuning** (GridSearchCV, RandomizedSearchCV)
5. **Use cross-validation** instead of single train-test split

### For Business Application:
1. Use model predictions to monitor customer satisfaction
2. Identify common complaints from negative reviews
3. Track sentiment trends over time
4. Alert management when negative sentiment increases
5. Use top features to guide product/service improvements

---

## � Deployment

### 🌐 Option 1: Streamlit Cloud (Recommended - FREE)

<div align="center">

**[🚀 LIVE DASHBOARD](https://sentimentanalysiscoffeeshop.streamlit.app)**

</div>

#### Deploy Steps (3 minutes):
```
1️⃣  Go to https://streamlit.io/cloud
2️⃣  Sign in with GitHub account (SeedFlora)
3️⃣  Click "New app"
4️⃣  Select repository: SeedFlora/sentimentAnalysisCoffeeShop
5️⃣  Branch: main
6️⃣  Main file path: streamlit_app.py
7️⃣  Click "Deploy" and wait 2-3 minutes
```

**Result:** App will be live at `https://sentimentanalysiscoffeeshop.streamlit.app`

### 💻 Option 2: Run Locally

```bash
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Install Streamlit
pip install streamlit

# 3. Run app
streamlit run streamlit_app.py

# 4. Open browser to http://localhost:8501
```

### 🎨 Dashboard Features

```
📊 Dashboard Utama
   ├── Overall sentiment statistics
   ├── Model performance metrics
   ├── Brand comparison charts
   └── Key insights summary

📈 Analisis Per Brand  
   ├── Brand selection dropdown
   ├── Sentiment distribution pie charts
   ├── Word cloud visualization
   ├── Performance metrics table
   └── Model comparison

🔮 Prediksi Sentimen
   ├── Real-time text input
   ├── Live sentiment prediction
   ├── Confidence score
   └── Top contributing words

📊 Perbandingan Model
   ├── All 6 models comparison
   ├── Metrics performance table
   ├── Ranking visualization
   └── Best model recommendation
```

---

## �📝 Method Summary

**Dataset**: 1,583 Indonesian reviews from 2 coffee shop chains  
**Preprocessing**: Text cleaning, TF-IDF feature extraction  
**Train-Test Split**: 80% (1,266) training, 20% (317) testing  
**Models Evaluated**: 6 machine learning algorithms  
**Best Performing**: Decision Tree (88.96% accuracy, 0.8694 F1-score)  
**Highest AUC-ROC**: Support Vector Machine (0.9046)  

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'nltk'"
```bash
pip install nltk --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
```

### Issue: Matplotlib showing empty plots
The script uses non-interactive backend ('Agg'). PNG files are saved to disk automatically.

### Issue: Script running too slow
- Reduce n_estimators in Random Forest and Gradient Boosting
- Reduce max_features in TfidfVectorizer
- Check available RAM and close other applications

### Issue: "NLTK stopwords not found"
The script automatically downloads stopwords. If not working:
```python
import nltk
nltk.download('stopwords')
```

---

## 🎓 Educational Value

This project demonstrates:
- ✅ **Data Preprocessing**: Text cleaning and normalization
- ✅ **Feature Engineering**: TF-IDF vectorization
- ✅ **Machine Learning**: 6 different algorithms
- ✅ **Model Evaluation**: Comprehensive metrics and visualization
- ✅ **Statistical Analysis**: Confusion matrices and ROC curves
- ✅ **NLP Basics**: Indonesian text processing and stopword removal
- ✅ **Software Engineering**: Virtual environments and dependency management

---

## 📚 References

### Papers & Concepts
- TF-IDF: Salton, G., & McGill, M. (1983). Introduction to Modern Information Retrieval.
- ROC Curves: Hanley, J. A., & McNeil, B. J. (1982). The meaning of area under a ROC curve.
- Confusion Matrix: Powers, D. M. (2011). Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness and Correlation.

### Python Libraries
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

---

## ✨ Future Enhancements

- [ ] Implement deep learning models (BERT, LSTM, GRU)
- [ ] Add aspect-based sentiment analysis (taste, service, ambience)
- [ ] Implement active learning for efficient data labeling
- [ ] Create REST API for real-time predictions
- [ ] Add multi-language support (English, Indonesian variants)
- [ ] Implement sentiment intensity prediction (not just binary)
- [ ] Deploy as web application (Flask/Streamlit)
- [ ] Add comparative analysis between chains

---

---

## 📊 Quick Stats

<div align="center">

```
🏆 MODEL PERFORMANCE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Best Accuracy:    88.96% (Decision Tree)
  Best AUC-ROC:     90.46% (SVM)
  Best F1-Score:    86.94% (Decision Tree)
  
📊 DATASET STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Reviews:    3,034
  Positive Reviews: 2,615 (86%)
  Negative Reviews:   419 (14%)
  Features (TF-IDF): 1,800
  
⏱️ PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Training Time:    ~2-3 minutes
  Inference Time:   <1 second per review
  Model Size:       ~5-50 MB each
```

</div>

---

## 👥 Contact & Support

<div align="center">

| Area | Details |
|------|---------|
| 📧 **Email** | juartobudi@gmail.com |
| 🐙 **GitHub** | [@SeedFlora](https://github.com/SeedFlora) |
| 🔗 **Repository** | [sentimentAnalysisCoffeeShop](https://github.com/SeedFlora/sentimentAnalysisCoffeeShop) |
| 🚀 **Live App** | [Streamlit Dashboard](https://sentimentanalysiscoffeeshop.streamlit.app) |

</div>

### 📞 Troubleshooting

<details>
<summary><b>Common Issues & Solutions</b> - Click to expand</summary>

#### ❌ ModuleNotFoundError
```bash
pip install nltk --trusted-host pypi.python.org
```

#### ❌ Matplotlib showing empty plots
Script uses non-interactive backend ('Agg'). PNG files saved automatically.

#### ❌ Script running too slow
- Reduce n_estimators in models
- Close other applications
- Check available RAM (4GB minimum)

#### ❌ NLTK stopwords not found
```python
import nltk
nltk.download('stopwords')
```

</details>

---

## 🙏 Acknowledgments

- ☕ **Kopi Nako** - Reviews dataset
- ☕ **Starbucks** - Reviews dataset  
- ☕ **Kopi Kenangan** - Reviews dataset
- 🇮🇩 Indonesian NLP community - Stopword resources
- 📦 Scikit-learn team - Excellent ML library
- 🎨 Streamlit team - Dashboard framework

---

## 📄 License

<div align="center">

**MIT License**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files.

[See LICENSE file for details](LICENSE)

</div>

---

## 📈 Project Stats

<div align="center">

![GitHub Repo Size](https://img.shields.io/github/repo-size/SeedFlora/sentimentAnalysisCoffeeShop?style=flat-square)
![GitHub Stars](https://img.shields.io/github/stars/SeedFlora/sentimentAnalysisCoffeeShop?style=flat-square)
![GitHub Forks](https://img.shields.io/github/forks/SeedFlora/sentimentAnalysisCoffeeShop?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/SeedFlora/sentimentAnalysisCoffeeShop?style=flat-square)

**Status**: ✅ Active & Maintained  
**Last Updated**: January 6, 2026  
**Version**: 1.0.0

</div>

---

<div align="center">

### 🌟 If you find this helpful, please consider giving it a star! ⭐

Made with ❤️ by Angel Donut Research Team

**Professional-grade machine learning implementation with comprehensive evaluation and production-ready dashboard.**

</div>
