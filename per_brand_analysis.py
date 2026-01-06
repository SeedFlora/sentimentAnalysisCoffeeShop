# -*- coding: utf-8 -*-
"""
Per-Brand Sentiment Analysis & Balancing
Analyzes Kopi Kenangan, Kopi Nako, and Starbucks separately
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PER-BRAND SENTIMENT ANALYSIS & DATA BALANCING")
print("="*80)

# ============================================================================
# PART 1: LOAD DATA FROM 3 BRANDS
# ============================================================================

print("\n[1] LOADING DATA FROM 3 BRANDS...")
print("-"*80)

# Load Kopi Nako
df_nako = pd.read_csv('kopinako_main_analysis.csv')
df_nako = df_nako.dropna(subset=['text', 'sentiment'])
df_nako = df_nako[df_nako['text'].str.len() > 0]
df_nako['brand'] = 'Kopi Nako'
print(f"✓ Kopi Nako: {len(df_nako)} reviews")

# Load Starbucks
df_starbucks = pd.read_csv('starbucks_detailed_reviews.csv')
df_starbucks = df_starbucks.dropna(subset=['text', 'sentiment_category'])
df_starbucks = df_starbucks[df_starbucks['text'].str.len() > 0]
df_starbucks = df_starbucks[df_starbucks['sentiment_category'].isin(['Positive', 'Negative'])]
df_starbucks = df_starbucks.rename(columns={'sentiment_category': 'sentiment'})
df_starbucks['sentiment'] = df_starbucks['sentiment'].map({'Positive': 'positive', 'Negative': 'negative'})
df_starbucks['brand'] = 'Starbucks'
print(f"✓ Starbucks: {len(df_starbucks)} reviews")

# Load Kopi Kenangan
try:
    df_kenangan = pd.read_excel('Kopi_Kenangan.xlsx')
    
    # Check columns - look for text and sentiment columns
    text_col = None
    sentiment_col = None
    
    for col in df_kenangan.columns:
        if 'review' in col.lower() or 'text' in col.lower() or 'description' in col.lower():
            text_col = col
        if 'sentiment' in col.lower() or 'opinion' in col.lower() or 'status' in col.lower():
            sentiment_col = col
    
    if text_col is None:
        # Try first column as text
        text_col = df_kenangan.columns[0]
    
    if sentiment_col is None:
        # Try to find sentiment from other columns
        print(f"  Available columns in Kopi_Kenangan.xlsx: {df_kenangan.columns.tolist()}")
        # If no sentiment column, we'll use the structure from the file
        # For now, assume we need to extract sentiment from comments or use all data as positive
        
    print(f"  Text column: {text_col}")
    if sentiment_col:
        print(f"  Sentiment column: {sentiment_col}")
    
    # Clean Kopi Kenangan data
    df_kenangan = df_kenangan.dropna(subset=[text_col])
    df_kenangan = df_kenangan[df_kenangan[text_col].astype(str).str.len() > 0]
    
    # Rename columns for consistency
    df_kenangan = df_kenangan.rename(columns={text_col: 'text'})
    
    if sentiment_col:
        df_kenangan = df_kenangan.rename(columns={sentiment_col: 'sentiment'})
        df_kenangan['sentiment'] = df_kenangan['sentiment'].astype(str).str.lower()
        # Map sentiments
        if 'negative' in df_kenangan['sentiment'].unique():
            pass  # Already has positive/negative
        else:
            # Try to map from other values
            print(f"  Sentiment values found: {df_kenangan['sentiment'].unique()[:5]}")
    else:
        # Assume all are positive if no sentiment column
        df_kenangan['sentiment'] = 'positive'
        print("  ⚠️  No sentiment column found - treating all as positive")
    
    df_kenangan['brand'] = 'Kopi Kenangan'
    print(f"✓ Kopi Kenangan: {len(df_kenangan)} reviews")
    
except Exception as e:
    print(f"⚠️  Could not load Kopi_Kenangan.xlsx: {e}")
    df_kenangan = None

# ============================================================================
# PART 2: ANALYZE BALANCE FOR EACH BRAND
# ============================================================================

print("\n[2] ANALYZING SENTIMENT BALANCE PER BRAND...")
print("-"*80)

brands_data = {
    'Kopi Nako': df_nako[['text', 'sentiment', 'brand']],
    'Starbucks': df_starbucks[['text', 'sentiment', 'brand']]
}

if df_kenangan is not None:
    brands_data['Kopi Kenangan'] = df_kenangan[['text', 'sentiment', 'brand']]

balance_info = {}

for brand_name, df_brand in brands_data.items():
    print(f"\n{brand_name}:")
    print("-" * 40)
    
    total = len(df_brand)
    sentiment_counts = df_brand['sentiment'].value_counts()
    
    print(f"Total reviews: {total}")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total) * 100
        print(f"  {sentiment}: {count:4d} ({percentage:6.2f}%)")
    
    # Check balance ratio
    if len(sentiment_counts) == 2:
        ratio = min(sentiment_counts) / max(sentiment_counts)
        balance_info[brand_name] = {
            'total': total,
            'counts': sentiment_counts,
            'ratio': ratio,
            'is_balanced': ratio > 0.4  # Consider balanced if ratio > 0.4 (40%)
        }
        
        if ratio > 0.4:
            status = "✅ BALANCED"
        else:
            status = "⚠️  IMBALANCED"
        
        print(f"Balance ratio: {ratio:.2%} {status}")
    else:
        balance_info[brand_name] = {
            'total': total,
            'counts': sentiment_counts,
            'ratio': None,
            'is_balanced': True
        }

# ============================================================================
# PART 3: VISUALIZATION - SENTIMENT DISTRIBUTION PER BRAND
# ============================================================================

print("\n[3] CREATING VISUALIZATIONS...")
print("-"*80)

# Plot 1: Sentiment Distribution per Brand (Before Balancing)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Sentiment Distribution per Brand (Before Balancing)', fontsize=14, fontweight='bold')

for idx, (brand_name, df_brand) in enumerate(brands_data.items()):
    ax = axes[idx]
    sentiment_counts = df_brand['sentiment'].value_counts()
    colors = ['#ff9999', '#66b3ff']
    
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors[:len(sentiment_counts)])
    ax.set_title(f'{brand_name}\n(Total: {len(df_brand)})', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10)
    ax.set_ylim(0, max(sentiment_counts.values) * 1.15)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add percentage
    total = len(df_brand)
    for i, (sentiment, count) in enumerate(sentiment_counts.items()):
        pct = (count / total) * 100
        ax.text(i, count/2, f'{pct:.1f}%', ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('brand_sentiment_before_balancing.png', dpi=300, bbox_inches='tight')
print("✓ Saved: brand_sentiment_before_balancing.png")
plt.close()

# ============================================================================
# PART 4: BALANCE IMBALANCED DATASETS
# ============================================================================

print("\n[4] BALANCING IMBALANCED DATASETS...")
print("-"*80)

balanced_data = {}

for brand_name, df_brand in brands_data.items():
    print(f"\n{brand_name}:")
    
    if balance_info[brand_name]['is_balanced']:
        print("  ✅ Already balanced - no action needed")
        balanced_data[brand_name] = df_brand.copy()
    else:
        print("  ⚠️  Imbalanced - applying balancing technique...")
        
        # Separate by sentiment
        df_positive = df_brand[df_brand['sentiment'] == 'positive']
        df_negative = df_brand[df_brand['sentiment'] == 'negative']
        
        # Get counts
        pos_count = len(df_positive)
        neg_count = len(df_negative)
        
        print(f"    Before: Positive={pos_count}, Negative={neg_count}")
        
        # Balance by upsampling minority class to match majority
        if pos_count > neg_count:
            # Oversample negative
            df_negative_balanced = resample(df_negative, 
                                           n_samples=pos_count,
                                           random_state=42,
                                           replace=True)
            df_balanced = pd.concat([df_positive, df_negative_balanced], ignore_index=True)
        else:
            # Oversample positive
            df_positive_balanced = resample(df_positive,
                                           n_samples=neg_count,
                                           random_state=42,
                                           replace=True)
            df_balanced = pd.concat([df_positive_balanced, df_negative], ignore_index=True)
        
        # Shuffle
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        balanced_data[brand_name] = df_balanced
        
        print(f"    After: Positive={len(df_balanced[df_balanced['sentiment']=='positive'])}, " +
              f"Negative={len(df_balanced[df_balanced['sentiment']=='negative'])}")

# ============================================================================
# PART 5: VISUALIZATION - AFTER BALANCING
# ============================================================================

print("\n[5] CREATING POST-BALANCING VISUALIZATIONS...")
print("-"*80)

# Plot 2: Before vs After Balancing (3-brand comparison)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Sentiment Distribution: Before vs After Balancing', fontsize=14, fontweight='bold')

for idx, (brand_name, df_brand) in enumerate(brands_data.items()):
    # BEFORE
    ax_before = axes[0, idx]
    sentiment_counts = df_brand['sentiment'].value_counts()
    colors = ['#ff9999', '#66b3ff']
    
    bars = ax_before.bar(sentiment_counts.index, sentiment_counts.values, color=colors[:len(sentiment_counts)])
    ax_before.set_title(f'{brand_name} - BEFORE', fontsize=11, fontweight='bold')
    ax_before.set_ylabel('Count', fontsize=10)
    ax_before.set_ylim(0, max(len(df_brand) * 0.9, max(sentiment_counts.values) * 1.15))
    
    for bar in bars:
        height = bar.get_height()
        ax_before.text(bar.get_x() + bar.get_width()/2., height,
                      f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # AFTER
    ax_after = axes[1, idx]
    df_balanced = balanced_data[brand_name]
    sentiment_counts_after = df_balanced['sentiment'].value_counts()
    
    bars = ax_after.bar(sentiment_counts_after.index, sentiment_counts_after.values, color=colors[:len(sentiment_counts_after)])
    ax_after.set_title(f'{brand_name} - AFTER', fontsize=11, fontweight='bold')
    ax_after.set_ylabel('Count', fontsize=10)
    ax_after.set_ylim(0, max(len(df_balanced) * 0.9, max(sentiment_counts_after.values) * 1.15))
    
    for bar in bars:
        height = bar.get_height()
        ax_after.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('brand_sentiment_before_after.png', dpi=300, bbox_inches='tight')
print("✓ Saved: brand_sentiment_before_after.png")
plt.close()

# ============================================================================
# PART 6: DETAILED COMPARISON TABLE
# ============================================================================

print("\n[6] DETAILED COMPARISON TABLE...")
print("-"*80)

# Create comparison dataframe
comparison_data = []

for brand_name, df_brand in brands_data.items():
    df_balanced = balanced_data[brand_name]
    
    # Before balancing
    pos_before = len(df_brand[df_brand['sentiment'] == 'positive'])
    neg_before = len(df_brand[df_brand['sentiment'] == 'negative'])
    total_before = len(df_brand)
    ratio_before = min(pos_before, neg_before) / max(pos_before, neg_before) if (pos_before > 0 and neg_before > 0) else 0
    
    # After balancing
    pos_after = len(df_balanced[df_balanced['sentiment'] == 'positive'])
    neg_after = len(df_balanced[df_balanced['sentiment'] == 'negative'])
    total_after = len(df_balanced)
    ratio_after = min(pos_after, neg_after) / max(pos_after, neg_after) if (pos_after > 0 and neg_after > 0) else 0
    
    comparison_data.append({
        'Brand': brand_name,
        'Before_Positive': pos_before,
        'Before_Negative': neg_before,
        'Before_Total': total_before,
        'Before_Ratio': f"{ratio_before:.2%}",
        'After_Positive': pos_after,
        'After_Negative': neg_after,
        'After_Total': total_after,
        'After_Ratio': f"{ratio_after:.2%}",
        'Status': '✅ Balanced' if ratio_after > 0.4 else '⚠️ Imbalanced'
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n" + comparison_df.to_string(index=False))

# Save to CSV
comparison_df.to_csv('brand_balance_comparison.csv', index=False)
print("\n✓ Saved: brand_balance_comparison.csv")

# ============================================================================
# PART 7: PIE CHARTS PER BRAND
# ============================================================================

print("\n[7] CREATING PIE CHARTS...")
print("-"*80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Sentiment Distribution Pie Charts: Before vs After Balancing', 
             fontsize=14, fontweight='bold')

colors = ['#ff9999', '#66b3ff']

for idx, (brand_name, df_brand) in enumerate(brands_data.items()):
    # BEFORE PIE
    ax_before = axes[0, idx]
    sentiment_counts = df_brand['sentiment'].value_counts()
    
    wedges, texts, autotexts = ax_before.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=colors[:len(sentiment_counts)],
        startangle=90
    )
    
    ax_before.set_title(f'{brand_name}\nBEFORE (n={len(df_brand)})', fontsize=11, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # AFTER PIE
    ax_after = axes[1, idx]
    df_balanced = balanced_data[brand_name]
    sentiment_counts_after = df_balanced['sentiment'].value_counts()
    
    wedges, texts, autotexts = ax_after.pie(
        sentiment_counts_after.values,
        labels=sentiment_counts_after.index,
        autopct='%1.1f%%',
        colors=colors[:len(sentiment_counts_after)],
        startangle=90
    )
    
    ax_after.set_title(f'{brand_name}\nAFTER (n={len(df_balanced)})', fontsize=11, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

plt.tight_layout()
plt.savefig('brand_sentiment_pie_charts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: brand_sentiment_pie_charts.png")
plt.close()

# ============================================================================
# PART 8: BALANCE RATIO COMPARISON
# ============================================================================

print("\n[8] CREATING BALANCE RATIO CHART...")
print("-"*80)

fig, ax = plt.subplots(figsize=(12, 6))

brands_list = list(brands_data.keys())
before_ratios = []
after_ratios = []

for brand_name, df_brand in brands_data.items():
    df_balanced = balanced_data[brand_name]
    
    pos_before = len(df_brand[df_brand['sentiment'] == 'positive'])
    neg_before = len(df_brand[df_brand['sentiment'] == 'negative'])
    ratio_before = min(pos_before, neg_before) / max(pos_before, neg_before) if (pos_before > 0 and neg_before > 0) else 0
    
    pos_after = len(df_balanced[df_balanced['sentiment'] == 'positive'])
    neg_after = len(df_balanced[df_balanced['sentiment'] == 'negative'])
    ratio_after = min(pos_after, neg_after) / max(pos_after, neg_after) if (pos_after > 0 and neg_after > 0) else 0
    
    before_ratios.append(ratio_before)
    after_ratios.append(ratio_after)

x = np.arange(len(brands_list))
width = 0.35

bars1 = ax.bar(x - width/2, before_ratios, width, label='Before Balancing', color='#ff9999', alpha=0.8)
bars2 = ax.bar(x + width/2, after_ratios, width, label='After Balancing', color='#66b3ff', alpha=0.8)

# Add threshold line
ax.axhline(y=0.4, color='red', linestyle='--', linewidth=2, label='Balance Threshold (40%)')

ax.set_xlabel('Brand', fontsize=12, fontweight='bold')
ax.set_ylabel('Balance Ratio (Min/Max)', fontsize=12, fontweight='bold')
ax.set_title('Class Balance Ratio: Before vs After Balancing', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(brands_list, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('brand_balance_ratio.png', dpi=300, bbox_inches='tight')
print("✓ Saved: brand_balance_ratio.png")
plt.close()

# ============================================================================
# PART 9: SAVE BALANCED DATASETS
# ============================================================================

print("\n[9] SAVING BALANCED DATASETS...")
print("-"*80)

for brand_name, df_balanced in balanced_data.items():
    filename = f"{brand_name.lower().replace(' ', '_')}_balanced.csv"
    df_balanced.to_csv(filename, index=False)
    print(f"✓ Saved: {filename} ({len(df_balanced)} samples)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n[FINAL SUMMARY]")
print("="*80)

for brand_name, df_brand in brands_data.items():
    df_balanced = balanced_data[brand_name]
    
    pos_before = len(df_brand[df_brand['sentiment'] == 'positive'])
    neg_before = len(df_brand[df_brand['sentiment'] == 'negative'])
    
    pos_after = len(df_balanced[df_balanced['sentiment'] == 'positive'])
    neg_after = len(df_balanced[df_balanced['sentiment'] == 'negative'])
    
    ratio_before = min(pos_before, neg_before) / max(pos_before, neg_before) if (pos_before > 0 and neg_before > 0) else 0
    ratio_after = min(pos_after, neg_after) / max(pos_after, neg_after) if (pos_after > 0 and neg_after > 0) else 0
    
    print(f"\n{brand_name}:")
    print(f"  Before: {pos_before} positive, {neg_before} negative (Ratio: {ratio_before:.1%})")
    print(f"  After:  {pos_after} positive, {neg_after} negative (Ratio: {ratio_after:.1%})")
    
    if ratio_after > 0.4:
        print(f"  Status: ✅ BALANCED")
    else:
        print(f"  Status: ⚠️  STILL IMBALANCED (consider collecting more data)")

print("\n[FILES GENERATED]")
print("  ✓ brand_sentiment_before_balancing.png")
print("  ✓ brand_sentiment_before_after.png")
print("  ✓ brand_sentiment_pie_charts.png")
print("  ✓ brand_balance_ratio.png")
print("  ✓ brand_balance_comparison.csv")
print("  ✓ kopi_nako_balanced.csv")
print("  ✓ starbucks_balanced.csv")
if df_kenangan is not None:
    print("  ✓ kopi_kenangan_balanced.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETED!")
print("="*80)
