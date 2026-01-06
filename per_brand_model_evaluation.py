"""
Per-Brand Model Evaluation
Mengevaluasi ML models untuk setiap brand terpisah
Menghitung: Accuracy, Precision, Recall, F1-Score
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
matplotlib.use('Agg')

print("\n" + "="*80)
print("[1] LOADING BALANCED DATA FROM 3 BRANDS...")
print("="*80 + "\n")

# Load balanced data
brands = {
    'Kopi Nako': 'kopi_nako_balanced.csv',
    'Starbucks': 'starbucks_balanced.csv',
    'Kopi Kenangan': 'kopi_kenangan_balanced.csv'
}

data = {}
for brand_name, filename in brands.items():
    df = pd.read_csv(filename)
    data[brand_name] = df
    print(f"✓ {brand_name}: {len(df)} samples")

print("\n" + "="*80)
print("[2] TEXT PREPROCESSING...")
print("="*80 + "\n")

def preprocess_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text.strip()

# Preprocess for each brand
for brand_name in data.keys():
    data[brand_name]['text_clean'] = data[brand_name]['text'].apply(preprocess_text)
    print(f"✓ {brand_name}: Text preprocessing complete")

print("\n" + "="*80)
print("[3] TRAINING MODELS FOR EACH BRAND...")
print("="*80 + "\n")

# Dictionary to store results
results = {
    'Brand': [],
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}

# Models to train
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
}

# Train and evaluate models for each brand
for brand_name, df in data.items():
    print(f"\n📊 PROCESSING: {brand_name}")
    print("-" * 60)
    
    # Prepare data
    X = df['text_clean']
    y = df['sentiment']
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2, max_df=0.8)
    X_tfidf = tfidf.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Train each model
    for model_name, model in models.items():
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Store results
            results['Brand'].append(brand_name)
            results['Model'].append(model_name)
            results['Accuracy'].append(round(accuracy * 100, 2))
            results['Precision'].append(round(precision * 100, 2))
            results['Recall'].append(round(recall * 100, 2))
            results['F1-Score'].append(round(f1 * 100, 2))
            
            print(f"  ✓ {model_name:20s} | Acc: {accuracy*100:6.2f}% | F1: {f1*100:6.2f}%")
        
        except Exception as e:
            print(f"  ✗ {model_name:20s} | Error: {str(e)[:40]}")

print("\n" + "="*80)
print("[4] SAVING RESULTS...")
print("="*80 + "\n")

# Create results dataframe
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('per_brand_evaluation_results.csv', index=False)
print(f"✓ Saved: per_brand_evaluation_results.csv")

# Display results by brand
print("\n" + "="*80)
print("[5] RESULTS SUMMARY BY BRAND")
print("="*80 + "\n")

for brand_name in brands.keys():
    brand_results = results_df[results_df['Brand'] == brand_name]
    print(f"\n📊 {brand_name.upper()}")
    print("-" * 80)
    
    # Best model
    best_model_idx = brand_results['F1-Score'].idxmax()
    best_model_row = brand_results.loc[best_model_idx]
    
    print(f"\n🏆 Best Model: {best_model_row['Model']}")
    print(f"   Accuracy:  {best_model_row['Accuracy']:6.2f}%")
    print(f"   Precision: {best_model_row['Precision']:6.2f}%")
    print(f"   Recall:    {best_model_row['Recall']:6.2f}%")
    print(f"   F1-Score:  {best_model_row['F1-Score']:6.2f}%")
    
    print(f"\nAll Models Performance:")
    print(brand_results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))

print("\n" + "="*80)
print("[6] CREATING VISUALIZATIONS...")
print("="*80 + "\n")

# Visualization 1: Comparison by Metric across Brands
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Per-Brand Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for metric, pos in zip(metrics, positions):
    ax = axes[pos]
    
    # Get best model for each brand
    best_per_brand = []
    brand_labels = []
    
    for brand_name in brands.keys():
        brand_data = results_df[results_df['Brand'] == brand_name]
        best_idx = brand_data['F1-Score'].idxmax()
        best_per_brand.append(brand_data.loc[best_idx, metric])
        brand_labels.append(brand_name)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(brand_labels, best_per_brand, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylim([0, 105])
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_title(f'{metric} (Best Model per Brand)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('per_brand_metrics_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: per_brand_metrics_comparison.png")
plt.close()

# Visualization 2: Heatmap of all metrics for best model per brand
fig, ax = plt.subplots(figsize=(10, 6))

heatmap_data = []
brand_labels = []

for brand_name in brands.keys():
    brand_data = results_df[results_df['Brand'] == brand_name]
    best_idx = brand_data['F1-Score'].idxmax()
    best_row = brand_data.loc[best_idx]
    
    heatmap_data.append([
        best_row['Accuracy'],
        best_row['Precision'],
        best_row['Recall'],
        best_row['F1-Score']
    ])
    brand_labels.append(f"{brand_name}\n({best_row['Model']})")

heatmap_array = np.array(heatmap_data)

sns.heatmap(heatmap_array, annot=True, fmt='.1f', cmap='RdYlGn', 
            xticklabels=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            yticklabels=brand_labels,
            cbar_kws={'label': 'Score (%)'}, ax=ax, vmin=70, vmax=100)

ax.set_title('Per-Brand Best Model Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('per_brand_metrics_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved: per_brand_metrics_heatmap.png")
plt.close()

# Visualization 3: Model comparison within each brand
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Model Performance Comparison within Each Brand', fontsize=14, fontweight='bold')

colors_models = plt.cm.Set3(np.linspace(0, 1, 6))

for idx, (brand_name, ax) in enumerate(zip(brands.keys(), axes)):
    brand_data = results_df[results_df['Brand'] == brand_name]
    
    x_pos = np.arange(len(brand_data))
    width = 0.2
    
    ax.bar(x_pos - 1.5*width, brand_data['Accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x_pos - 0.5*width, brand_data['Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x_pos + 0.5*width, brand_data['Recall'], width, label='Recall', alpha=0.8)
    ax.bar(x_pos + 1.5*width, brand_data['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title(brand_name, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(brand_data['Model'], rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 105])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('per_brand_models_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: per_brand_models_comparison.png")
plt.close()

# Visualization 4: F1-Score Radar Chart comparison
from math import pi

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

categories = list(brands.keys())
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

for idx, model_name in enumerate(list(models.keys())[:4]):  # First 4 models for clarity
    values = []
    for brand_name in brands.keys():
        brand_data = results_df[(results_df['Brand'] == brand_name) & (results_df['Model'] == model_name)]
        if len(brand_data) > 0:
            values.append(brand_data['F1-Score'].values[0])
        else:
            values.append(0)
    
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
ax.set_title('F1-Score Radar Chart - 4 Best Models', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('per_brand_f1_radar.png', dpi=150, bbox_inches='tight')
print("✓ Saved: per_brand_f1_radar.png")
plt.close()

print("\n" + "="*80)
print("[7] DETAILED RESULTS TABLE")
print("="*80 + "\n")

# Create summary table
summary_table = results_df.pivot_table(
    index='Brand',
    columns='Model',
    values='F1-Score',
    aggfunc='first'
)

print("\n📊 F1-Score Comparison Table (%):\n")
print(summary_table.round(2).to_string())

print("\n" + "="*80)
print("[8] ANALYSIS COMPLETE!")
print("="*80)

print("\n✅ Generated Files:")
print("   • per_brand_evaluation_results.csv")
print("   • per_brand_metrics_comparison.png")
print("   • per_brand_metrics_heatmap.png")
print("   • per_brand_models_comparison.png")
print("   • per_brand_f1_radar.png")

print("\n" + "="*80 + "\n")
