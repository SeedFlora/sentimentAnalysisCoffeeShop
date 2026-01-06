# -*- coding: utf-8 -*-
"""
Sentiment Analysis with Multiple ML Models
Analyzes reviews from Kopi Nako and Starbucks datasets
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)
import re
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Download stopwords if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("="*80)
print("SENTIMENT ANALYSIS MACHINE LEARNING PROJECT")
print("="*80)

# ============================================================================
# PART 1: LOAD AND EXPLORE DATA
# ============================================================================

print("\n[1] LOADING DATA...")
print("-"*80)

# Load Kopi Nako data
df_kopi = pd.read_csv('kopinako_main_analysis.csv')
print(f"Kopi Nako dataset: {df_kopi.shape}")
print(f"  Columns: {df_kopi.columns.tolist()}")
print(f"  Sentiment values: {df_kopi['sentiment'].unique()}")

# Load Starbucks data
df_starbucks = pd.read_csv('starbucks_detailed_reviews.csv')
print(f"\nStarbucks dataset: {df_starbucks.shape}")
print(f"  Columns: {df_starbucks.columns.tolist()}")
print(f"  Sentiment values: {df_starbucks['sentiment_category'].unique()}")

# ============================================================================
# PART 2: DATA CLEANING AND PREPROCESSING
# ============================================================================

print("\n[2] DATA CLEANING AND PREPROCESSING...")
print("-"*80)

# Clean Kopi Nako data
df_kopi = df_kopi.dropna(subset=['text', 'sentiment'])
df_kopi = df_kopi[df_kopi['text'].str.len() > 0]
print(f"Kopi Nako after cleaning: {df_kopi.shape}")

# Clean Starbucks data
df_starbucks = df_starbucks.dropna(subset=['text', 'sentiment_category'])
df_starbucks = df_starbucks[df_starbucks['text'].str.len() > 0]
# Remove non-sentiment rows
df_starbucks = df_starbucks[df_starbucks['sentiment_category'].isin(['Positive', 'Negative'])]
print(f"Starbucks after cleaning: {df_starbucks.shape}")

# Rename sentiment column for consistency
df_starbucks = df_starbucks.rename(columns={'sentiment_category': 'sentiment'})
# Map sentiment values to lowercase
df_starbucks['sentiment'] = df_starbucks['sentiment'].map({'Positive': 'positive', 'Negative': 'negative'})

# Combine both datasets
df = pd.concat([df_kopi[['text', 'sentiment']], 
                df_starbucks[['text', 'sentiment']]], 
               ignore_index=True)

# Remove any remaining NaN values
df = df.dropna(subset=['text', 'sentiment'])
df = df[df['text'].str.len() > 0]

print(f"\nCombined dataset: {df.shape}")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
print(f"Sentiment percentage:\n{df['sentiment'].value_counts(normalize=True) * 100}")

# ============================================================================
# PART 3: TEXT PREPROCESSING
# ============================================================================

print("\n[3] TEXT PREPROCESSING...")
print("-"*80)

def preprocess_text(text):
    """Clean and preprocess text"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    return ''

df['cleaned_text'] = df['text'].apply(preprocess_text)

print(f"Sample preprocessed texts:")
for i in range(min(3, len(df))):
    print(f"  {i+1}. Original: {df['text'].iloc[i][:80]}...")
    print(f"     Cleaned:  {df['cleaned_text'].iloc[i][:80]}...")

# ============================================================================
# PART 4: FEATURE ENGINEERING
# ============================================================================

print("\n[4] FEATURE ENGINEERING (TF-IDF)...")
print("-"*80)

# Encode sentiment labels
le = LabelEncoder()
df['sentiment_encoded'] = le.fit_transform(df['sentiment'])

print(f"Sentiment encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Prepare features and target
X = df['cleaned_text']
y = df['sentiment_encoded']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set sentiment distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Test set sentiment distribution:\n{pd.Series(y_test).value_counts()}")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=5000, 
    stop_words=stopwords.words('indonesian'),
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"\nTF-IDF Matrix shape:")
print(f"  Training: {X_train_tfidf.shape}")
print(f"  Test: {X_test_tfidf.shape}")
print(f"  Features extracted: {len(tfidf.get_feature_names_out())}")

# ============================================================================
# PART 5: MODEL TRAINING AND EVALUATION
# ============================================================================

print("\n[5] TRAINING MACHINE LEARNING MODELS...")
print("-"*80)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs'),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=80, random_state=42, max_depth=15),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=80, random_state=42, max_depth=5)
}

# Store results
results = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    try:
        # Handle sparse matrices
        if name in ['Gradient Boosting', 'AdaBoost', 'Decision Tree', 'Random Forest']:
            model.fit(X_train_tfidf.toarray(), y_train)
            y_pred = model.predict(X_test_tfidf.toarray())
            y_pred_proba = model.predict_proba(X_test_tfidf.toarray())
        else:
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_tfidf)
            else:
                y_pred_proba = None
        
        predictions[name] = y_pred
        probabilities[name] = y_pred_proba
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate AUC-ROC for binary classification
        if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
            auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc_roc = None
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc
        }
        
        print(f"  ✓ Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        if auc_roc is not None:
            print(f"    AUC-ROC: {auc_roc:.4f}")
    
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

# ============================================================================
# PART 6: RESULTS COMPARISON
# ============================================================================

print("\n[6] MODEL PERFORMANCE COMPARISON")
print("="*80)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('f1_score', ascending=False)

print("\nPerformance Metrics Summary:")
print(results_df.round(4))

# Save results to CSV
results_df.to_csv('model_performance_results.csv')
print(f"\n✓ Results saved to 'model_performance_results.csv'")

# ============================================================================
# PART 7: DETAILED EVALUATION METRICS
# ============================================================================

print("\n[7] DETAILED EVALUATION FOR TOP 3 MODELS")
print("="*80)

top_models = results_df.head(3).index.tolist()

for model_name in top_models:
    print(f"\n{model_name}")
    print("-"*80)
    
    y_pred = predictions[model_name]
    cm = confusion_matrix(y_test, y_pred)
    
    # Print confusion matrix
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nConfusion Matrix breakdown:")
    print(f"  True Negatives (TN):  {cm[0, 0]}")
    print(f"  False Positives (FP): {cm[0, 1]}")
    print(f"  False Negatives (FN): {cm[1, 0]}")
    print(f"  True Positives (TP):  {cm[1, 1]}")
    
    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Print AUC-ROC if available
    if probabilities[model_name] is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, probabilities[model_name][:, 1])
        auc_score = auc(fpr, tpr)
        print(f"AUC-ROC Score: {auc_score:.4f}")

# ============================================================================
# PART 8: VISUALIZATIONS
# ============================================================================

print("\n[8] GENERATING VISUALIZATIONS...")
print("-"*80)

# Plot 1: Model Performance Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy comparison
ax = axes[0, 0]
models_sorted = results_df.index
accuracies = results_df['accuracy']
bars = ax.barh(models_sorted, accuracies, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
ax.set_xlabel('Accuracy', fontsize=12)
ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{acc:.4f}', ha='left', va='center', fontsize=10)

# All metrics comparison
ax = axes[0, 1]
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
x_pos = np.arange(len(models_sorted))
width = 0.2

for i, metric in enumerate(metrics):
    values = results_df[metric].fillna(0)
    ax.bar(x_pos + i*width, values, width, label=metric, alpha=0.8)

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Scores', fontsize=12)
ax.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos + width*1.5)
ax.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=10)
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

# Confusion matrices for top 2 models
for idx, model_name in enumerate(top_models[:2]):
    ax = axes[1, idx]
    y_pred = predictions[model_name]
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=ax, cbar=True)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison.png")
plt.show()

# Plot 2: ROC Curves for Top Model
fig, ax = plt.subplots(figsize=(10, 8))

best_model_name = top_models[0]
if probabilities[best_model_name] is not None and len(np.unique(y_test)) == 2:
    fpr, tpr, _ = roc_curve(y_test, probabilities[best_model_name][:, 1])
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: roc_curve.png")
    plt.show()

# Plot 3: Sentiment Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Overall distribution
sentiment_counts = df['sentiment'].value_counts()
axes[0].bar(sentiment_counts.index, sentiment_counts.values, color=['#ff9999', '#66b3ff'])
axes[0].set_title('Sentiment Distribution (All Data)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# Train-Test distribution
train_sentiment = pd.Series(y_train).map({0: le.classes_[0], 1: le.classes_[1]}).value_counts()
test_sentiment = pd.Series(y_test).map({0: le.classes_[0], 1: le.classes_[1]}).value_counts()

x = np.arange(len(le.classes_))
width = 0.35

axes[1].bar(x - width/2, [train_sentiment.get(c, 0) for c in le.classes_], 
            width, label='Train', alpha=0.8)
axes[1].bar(x + width/2, [test_sentiment.get(c, 0) for c in le.classes_], 
            width, label='Test', alpha=0.8)
axes[1].set_title('Train-Test Sentiment Distribution', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_xticks(x)
axes[1].set_xticklabels(le.classes_)
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sentiment_distribution.png")
plt.show()

# ============================================================================
# PART 9: FEATURE IMPORTANCE
# ============================================================================

print("\n[9] FEATURE IMPORTANCE ANALYSIS")
print("-"*80)

feature_names = tfidf.get_feature_names_out()

# Random Forest Feature Importance
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-15:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color='skyblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel('Relative Importance', fontsize=11)
    ax.set_title('Random Forest - Top 15 Important Features', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_importance_rf.png")
    plt.show()

# Logistic Regression Coefficients
if 'Logistic Regression' in models:
    lr_model = models['Logistic Regression']
    coefficients = lr_model.coef_[0]
    indices = np.argsort(np.abs(coefficients))[-15:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red' if coef < 0 else 'blue' for coef in coefficients[indices]]
    ax.barh(range(len(indices)), coefficients[indices], color=colors, alpha=0.7)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel('Coefficient Value', fontsize=11)
    ax.set_title('Logistic Regression - Top 15 Feature Coefficients', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance_lr.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_importance_lr.png")
    plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n[FINAL SUMMARY]")
print("="*80)
print(f"Total models trained: {len(models)}")
print(f"Best model: {results_df.index[0]} (F1-Score: {results_df.iloc[0]['f1_score']:.4f})")
print(f"Best accuracy: {results_df.iloc[0]['accuracy']:.4f}")
print(f"\nDataset Information:")
print(f"  Total samples: {len(df)}")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"\nSentiment Distribution:")
for sentiment, count in df['sentiment'].value_counts().items():
    percentage = (count / len(df)) * 100
    print(f"  {sentiment}: {count} samples ({percentage:.1f}%)")

print("\n[FILES GENERATED]")
print("  ✓ model_performance_results.csv")
print("  ✓ model_comparison.png")
print("  ✓ roc_curve.png")
print("  ✓ sentiment_distribution.png")
print("  ✓ feature_importance_rf.png")
print("  ✓ feature_importance_lr.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETED!")
print("="*80)
