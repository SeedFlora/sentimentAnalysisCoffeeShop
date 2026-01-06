# Usage Guide & Practical Examples

## Quick Start

### Run Full Analysis (Default)
```bash
python sentiment_analysis.py
```

This will:
1. ✅ Load both CSV datasets
2. ✅ Clean and preprocess reviews
3. ✅ Train 6 machine learning models
4. ✅ Evaluate with comprehensive metrics
5. ✅ Generate 6 visualization files
6. ✅ Save model comparison to CSV

**Runtime**: ~2-3 minutes (varies by system)

---

## Understanding the Output Files

### 1. model_performance_results.csv

**What it is**: Performance comparison of all trained models

**How to read it**:
```
,accuracy,precision,recall,f1_score,auc_roc
Decision Tree,0.8896,0.8756,0.8896,0.8694,0.7084
...
```

**Column meanings**:
- **accuracy**: Overall correctness (higher is better)
- **precision**: Accuracy of positive predictions
- **recall**: Ability to find positive instances
- **f1_score**: Balance between precision & recall
- **auc_roc**: Discrimination ability (0.5=random, 1.0=perfect)

**How to use it**:
- Sort by f1_score to find best balanced model
- Check auc_roc for robust probability estimates
- Compare your baseline against these scores

---

### 2. model_comparison.png

**Contains**: 4 subplots
1. **Top-left**: Accuracy comparison bar chart
   - Decision Tree leads with ~89%
   
2. **Top-right**: All metrics comparison (grouped bars)
   - See trade-offs between precision/recall
   
3. **Bottom-left**: Confusion matrix for Decision Tree
   - Green = correct predictions
   - Red = incorrect predictions
   
4. **Bottom-right**: Confusion matrix for Gradient Boosting
   - Alternative best model

**How to interpret**:
- Higher bars = better performance
- Balanced colored boxes = good model

---

### 3. roc_curve.png

**What it shows**: ROC curve for best AUC model (Support Vector Machine)

**Understanding ROC Curve**:
- X-axis: False Positive Rate (0 to 1)
- Y-axis: True Positive Rate (0 to 1)
- Diagonal line: Random classifier (no skill)
- Orange curve: Your model
- Higher curve = better model
- AUC = 0.9046 (excellent!)

**How to use it**:
- If curve close to diagonal = model not generalizing
- If curve in top-left corner = excellent discrimination
- Useful for setting classification thresholds

---

### 4. sentiment_distribution.png

**Contains**: 2 charts
1. **Left**: Overall sentiment distribution
   - Shows data imbalance (86% positive, 14% negative)
   
2. **Right**: Train-test sentiment split
   - Ensures both sets have similar distribution

**Why it matters**:
- Class imbalance affects model training
- Good stratified split prevents bias
- Imbalance suggests using weighted loss or SMOTE

---

### 5. feature_importance_rf.png

**Shows**: Top 15 features from Random Forest model

**Reading it**:
- X-axis: Importance value (length of bar)
- Y-axis: Feature (word/term)
- Longer bars = more important for predictions

**Top features might include**:
- "enak" (delicious) - positive indicator
- "juara" (excellent) - positive indicator
- "nyaman" (comfortable) - positive indicator
- "ramai" (crowded) - negative indicator

**Business use**:
- Focus on improving top negative indicators
- Highlight top positive features in marketing
- Track mentions of important words over time

---

### 6. feature_importance_lr.png

**Shows**: Top 15 coefficients from Logistic Regression

**Color coding**:
- 🔵 **Blue bars**: Positive sentiment indicators (push score up)
- 🔴 **Red bars**: Negative sentiment indicators (push score down)
- **Longer bars**: Stronger influence

**Interpretation**:
- Magnitude shows strength of association
- Direction (red/blue) shows sentiment direction
- More interpretable than Random Forest importance

---

## Common Questions

### Q: Which model should I use in production?
**A**: Use **Support Vector Machine** (best AUC-ROC: 0.9046)
- Most reliable probability estimates
- Best generalization to new data
- If speed matters: use **Decision Tree**
- If balanced performance: use **Gradient Boosting**

### Q: Why is accuracy not the best metric?
**A**: With imbalanced data (86% positive):
- A model that predicts "always positive" gets 86% accuracy
- But F1-score and AUC-ROC show real performance
- Use **F1-Score** for balanced performance
- Use **AUC-ROC** for probability ranking

### Q: What does AUC-ROC = 0.9046 mean?
**A**: The model has 90.46% probability of:
- Ranking a random positive review higher than a random negative review
- Better at distinguishing between classes

### Q: My accuracy is 88.96% - is that good?
**A**: Yes! In context:
- Baseline (always predict positive) = 86.3%
- Model improvement = 2.66%
- But F1-score (0.8694) is more reliable metric

### Q: Why are False Negatives low but False Positives high?
**A**: Model is conservative:
- Rarely says "negative" (conservative)
- Sometimes says "positive" when actually "negative"
- Trade-off: catches more positives but over-predicts

### Q: Can I improve these results?
**A**: Yes! Try:
1. Collect more negative examples
2. Use ensemble voting of top 3 models
3. Implement deep learning (BERT, LSTM)
4. Apply hyperparameter tuning
5. Use cross-validation

---

## Practical Applications

### 1. Sentiment Monitoring Dashboard
```python
# Pseudo-code
for new_review in daily_reviews:
    sentiment = model.predict(new_review)
    if sentiment == 'negative':
        alert_management()
        add_to_issue_tracker()
```

### 2. Customer Feedback Analysis
```
Positive indicators to promote:
- "enak" (delicious)
- "baik" (good)
- "ramah" (friendly)

Negative issues to fix:
- "ramai" (crowded)
- "lama" (slow service)
- "mahal" (expensive)
```

### 3. Competitive Analysis
```
Compare Kopi Nako vs Starbucks:
- Which has higher positive sentiment?
- What are unique complaints for each?
- Which service aspects score better?
```

### 4. Time Series Tracking
```
Track sentiment over time:
- Monthly average positive %
- Identify trend changes
- Correlate with operational changes
```

---

## Modifying the Script

### Change Test Set Size
```python
# Original: 80-20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # ← Change this to 0.3 for 70-30 split
    ...
)
```

### Add New Models
```python
from sklearn.linear_model import SGDClassifier

models = {
    'Logistic Regression': LogisticRegression(...),
    'Multinomial Naive Bayes': MultinomialNB(),
    'SGD Classifier': SGDClassifier(),  # ← Add new model
    ...
}
```

### Change Features
```python
# Original: 1800 features
tfidf = TfidfVectorizer(
    max_features=5000,  # ← Increase to 5000
    ngram_range=(1, 2),  # ← Change to (1, 3) for 3-grams
    ...
)
```

### Change Random Seed (reproducibility)
```python
# Change random_state from 42 to any other number
random_state=42  # ← Change this
```

---

## Performance Optimization Tips

### Speed Up Training
1. Reduce max_features in TfidfVectorizer (e.g., 3000 → 1000)
2. Reduce n_estimators in Random Forest (e.g., 100 → 50)
3. Use smaller max_depth in tree-based models
4. Increase test_size (e.g., 0.2 → 0.3) to reduce training data

### Improve Accuracy
1. Increase features: max_features in TfidfVectorizer
2. Add n-grams: ngram_range=(1, 3)
3. Tune hyperparameters using GridSearchCV
4. Collect more training data
5. Try ensemble methods combining multiple models

### Reduce Overfitting
1. Increase max_df in TfidfVectorizer
2. Decrease max_depth in tree-based models
3. Use cross-validation
4. Add regularization (L1/L2)

---

## Integration with Your Application

### Using the Trained Model

```python
# Save model after training
import pickle

with open('best_model.pkl', 'wb') as f:
    pickle.dump(models['Decision Tree'], f)

# Load in production
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocess new review
new_review = "Kopi enak dan ramah service"
cleaned = preprocess_text(new_review)
features = tfidf.transform([cleaned])

# Make prediction
sentiment = model.predict(features)[0]
probability = model.predict_proba(features)[0]

print(f"Sentiment: {sentiment}")  # 0 or 1
print(f"Probability: {probability}")  # [neg_prob, pos_prob]
```

### Creating REST API (Flask)

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    
    # Preprocess
    cleaned = preprocess_text(review)
    features = tfidf.transform([cleaned])
    
    # Predict
    sentiment = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return jsonify({
        'sentiment': 'positive' if sentiment == 1 else 'negative',
        'confidence': float(max(probability))
    })

if __name__ == '__main__':
    app.run(debug=True)
```

---

## Monitoring Model Performance

### Regular Evaluation
```python
# Every month, evaluate on new data
new_accuracy = model.score(X_new, y_new)
if new_accuracy < previous_accuracy - 0.05:  # 5% drop
    alert('Model performance degraded!')
    retrain_model()
```

### Track Predictions
```python
# Log all predictions for analysis
prediction_log = {
    'date': datetime.now(),
    'review': review_text,
    'prediction': sentiment,
    'confidence': confidence,
    'actual': actual_sentiment  # After human verification
}
```

---

## Troubleshooting Results

### Problem: Model predicts everything as positive

**Possible causes**:
1. Data imbalance (86% positive)
2. Model too simple or undertrained
3. Features not discriminative

**Solutions**:
1. Use class_weight='balanced' in model
2. Apply SMOTE for oversampling negatives
3. Use more complex model
4. Increase features (ngram_range)

### Problem: Very low recall on negative class

**Possible causes**:
1. Insufficient negative training examples
2. Feature overlap between classes
3. Model biased toward majority class

**Solutions**:
1. Oversample negative examples (SMOTE)
2. Use F1-score instead of accuracy
3. Adjust classification threshold
4. Use weighted loss functions

### Problem: Overfitting (high train accuracy, low test accuracy)

**Possible causes**:
1. Model too complex
2. Insufficient regularization
3. Too many features
4. Small test set

**Solutions**:
1. Reduce max_depth or n_estimators
2. Increase max_df to filter common terms
3. Use cross-validation
4. Simplify features

---

## Next Steps

1. **Explore**: Examine top features in detail
2. **Experiment**: Try different hyperparameters
3. **Integrate**: Add predictions to your application
4. **Monitor**: Track performance over time
5. **Improve**: Collect more data and retrain
6. **Deploy**: Make model available to users

---

## Resources

- [Scikit-learn Guide](https://scikit-learn.org/stable/user_guide.html)
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

---

**Happy analyzing! 🚀**
