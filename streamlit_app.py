"""
Sentiment Analysis Dashboard
Coffee Shops Review Sentiment Analysis with Interactive Features
Deployed with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib
import warnings
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import os
import re

warnings.filterwarnings('ignore')

# Indonesian stopwords to filter out - BASED ON CORPUS ANALYSIS
INDONESIAN_STOPWORDS = {
    # TOP FREQUENT NON-MEANINGFUL (dari analisis corpus)
    'nya', 'di', 'dan', 'yg', 'yang', 'ada', 'saya', 'tidak', 'untuk', 'juga',
    'sama', 'tapi', 'ga', 'lagi', 'the', 'ini', 'bisa', 'dari', 'saja', 'jadi',
    'kalau', 'sih', 'kalo', 'dengan', 'aja', 'ke', 'gak', 'cuma', 'pas', 'sekali',
    'udah', 'hanya', 'karena', 'bgt', 'malah', 'padahal', 'itu', 'agak', 'orang',
    'sudah', 'lah', 'to', 'cukup', 'terlalu', 'jam', 'and', 'atau', 'of', 'kali',
    'tp', 'jg', 'tdk', 'gk', 'dah', 'nih', 'tuh', 'dong', 'deh', 'sih', 'ya',
    'kan', 'kok', 'lho', 'nah', 'pun', 'pula', 'banget', 'bgt', 'sangat', 'sekali',
    
    # Brand names (tidak informatif untuk sentiment)
    'kopi', 'kenangan', 'starbuck', 'starbucks', 'nako', 'kopken', 'coffee', 'cafe',
    'kopinako', 'kopikenangan', 'starbuckscoffee',
    
    # Common location/place words
    'tempat', 'tempatnya', 'disini', 'disana', 'disitu', 'place', 'mall', 'area',
    'sini', 'sana', 'situ',
    
    # Common verbs (terlalu umum)
    'buat', 'mau', 'beli', 'pesan', 'order', 'bikin', 'minum', 'makan', 'suka',
    'harus', 'tolong', 'kasih', 'ambil', 'datang', 'pergi', 'pulang', 'kerja',
    'pake', 'pakai', 'coba', 'cobain', 'kesini', 'kesana',
    
    # Common adjectives (terlalu umum)  
    'enak', 'nyaman', 'bagus', 'baik', 'good', 'nice', 'luas', 'bersih', 'cocok',
    'oke', 'ok', 'best', 'great', 'lama', 'kurang', 'lebih', 'sangat', 'banget',
    'mantap', 'mantab', 'hebat', 'keren', 'asik', 'asyik',
    
    # Common nouns (tidak spesifik sentiment)
    'pelayanan', 'harga', 'minuman', 'makanan', 'barista', 'kasir', 'menu', 'rasa',
    'parkir', 'tukang', 'ngopi', 'nongkrong', 'banyak', 'pelayan', 'service',
    
    # English common words
    'the', 'and', 'to', 'of', 'is', 'it', 'in', 'for', 'on', 'with', 'this', 'that',
    'are', 'was', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can', 'very', 'really', 'always',
    'place', 'one', 'love', 'like', 'so', 'but', 'just', 'get', 'go', 'come',
    
    # Pronoun
    'kami', 'dia', 'mereka', 'kamu', 'mu', 'ku', 'ane', 'gue', 'gw', 'elo', 'lo',
    'anda', 'kita', 'kalian', 'beliau',
    
    # Nama/Title
    'pak', 'bu', 'mas', 'mbak', 'bro', 'kak', 'bang', 'om', 'tante', 'dik', 'bos',
    
    # Lainnya - kata tidak baku/singkatan
    'yaa', 'yahh', 'yah', 'yg', 'dgn', 'utk', 'krn', 'sm', 'lg', 'aj', 'bngt',
    'bgt', 'gt', 'gtu', 'gni', 'gmn', 'gmna', 'emg', 'emang', 'mmg', 'memang',
    'kyk', 'kayak', 'kek', 'trs', 'terus', 'trus', 'abis', 'habis', 'udh', 'sdh',
    'blm', 'belum', 'msh', 'masih', 'bs', 'gbs', 'gabisa', 'gaada', 'gada',
    'dr', 'pd', 'dl', 'dulu', 'skrg', 'sekarang', 'kmrn', 'kemarin', 'bsk', 'besok',
    'org', 'ornag', 'tmpt', 'tmpat', 'bkn', 'bukan', 'krg', 'kurang',
    'dst', 'dll', 'dsb', 'etc', 'waktu', 'hari', 'menit', 'detik',
}

# Combine dengan English stopwords
COMBINED_STOPWORDS = STOPWORDS.union(INDONESIAN_STOPWORDS)

def clean_text_for_wordcloud(text):
    """Clean text for wordcloud - remove non-meaningful words"""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove email
    text = re.sub(r'\S+@\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove special characters except spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove single characters
    text = ' '.join([word for word in text.split() if len(word) > 2])
    return text
    'aja', 'kan', 'lah', 'deh', 'yah', 'ya', 'sih', 'dong', 'nah', 'lho', 'kok',
    'pun', 'pula', 'tuh', 'nih', 'gak', 'tidak', 'jg', 'tdk', 'hrs', 'bgt',
    
    # Nama/Title umum
    'pak', 'bu', 'mas', 'mbak', 'bro', 'kak', 'adik', 'kakak', 'abang', 'bang',
    'om', 'tante', 'nenek', 'kakek', 'dik', 'bos', 'boss',
    
    # Common review words yang tidak informatif
    'saya', 'kali', 'kalo', 'kalau', 'gimana', 'gimn', 'gini', 'gitu', 'dst',
    'dll', 'dsb', 'etc', 'tapi', 'tp', 'pdhal', 'padhal', 'meski', 'meskipun',
    'karena', 'soalnya', 'makanya', 'eh', 'oh', 'ah', 'wow', 'wah', 'duh',
}

# Combine with English stopwords
COMBINED_STOPWORDS = STOPWORDS.union(INDONESIAN_STOPWORDS)

# Set page config
st.set_page_config(
    page_title="☕ Coffee Sentiment Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 0rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .positive-color {
        color: #2ecc71;
        font-weight: bold;
    }
    .negative-color {
        color: #e74c3c;
        font-weight: bold;
    }
    .neutral-color {
        color: #3498db;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_resource
def load_data():
    """Load all datasets"""
    nako = pd.read_csv('kopi_nako_balanced.csv')
    starbucks = pd.read_csv('starbucks_balanced.csv')
    kenangan = pd.read_csv('kopi_kenangan_balanced.csv')
    metrics = pd.read_csv('per_brand_evaluation_results.csv')
    return nako, starbucks, kenangan, metrics

@st.cache_resource
def train_models():
    """Train and cache models"""
    nako, starbucks, kenangan, _ = load_data()
    
    models = {}
    vectorizers = {}
    
    brands_data = {
        'Kopi Nako': (nako, SVC(kernel='linear', random_state=42)),
        'Starbucks': (kenangan, GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)),
        'Kopi Kenangan': (kenangan, SVC(kernel='linear', random_state=42))
    }
    
    for brand_name, (df, model) in brands_data.items():
        # Preprocess
        X = df['text'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
        y = df['sentiment']
        
        # Vectorize
        vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2, max_df=0.8)
        X_tfidf = vec.fit_transform(X)
        
        # Train
        model.fit(X_tfidf, y)
        
        models[brand_name] = model
        vectorizers[brand_name] = vec
    
    return models, vectorizers

@st.cache_data
def get_wordcloud_data(df, sentiment):
    """Generate wordcloud data"""
    text = ' '.join(df[df['sentiment'] == sentiment]['text'].values)
    return text

# ============================================================================
# MAIN APP
# ============================================================================

# Load data
nako, starbucks, kenangan, metrics = load_data()

# Sidebar Navigation
st.sidebar.title("☕ Navigation")
page = st.sidebar.radio(
    "Pilih halaman:",
    ["🏠 Dashboard Utama", "📊 Analisis Per Brand", "🔮 Prediksi Sentiment", "📈 Perbandingan Model"]
)

# ============================================================================
# PAGE 1: DASHBOARD UTAMA
# ============================================================================

if page == "🏠 Dashboard Utama":
    st.title("☕ Sentiment Analysis Dashboard")
    st.markdown("### Analisis Sentimen Review Kopi Indonesia")
    
    st.markdown("""
    ---
    #### 📌 Ringkasan Proyek
    
    Proyek ini menganalisis sentimen pelanggan terhadap **3 brand kopi ternama Indonesia**:
    - ☕ **Kopi Nako** - 1,768 reviews
    - ☕ **Starbucks** - 964 reviews  
    - ☕ **Kopi Kenangan** - 2,518 reviews
    
    **Total: 5,250 reviews** yang sudah di-balance (50% positive, 50% negative)
    
    Menggunakan **6 ML models** untuk klasifikasi sentiment:
    1. Support Vector Machine (SVM) - **Best Performance**
    2. Gradient Boosting
    3. Logistic Regression
    4. Random Forest
    5. Naive Bayes
    6. Decision Tree
    """)
    
    # Key Metrics
    st.markdown("---")
    st.markdown("#### 🏆 Best Model Performance Per Brand")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown("##### ☕ Kopi Nako")
        st.markdown('<p class="positive-color" style="font-size: 28px;">97.74%</p>', unsafe_allow_html=True)
        st.markdown("**F1-Score (SVM)**")
        st.markdown("Accuracy: 97.74% | Precision: 97.84%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown("##### ☕ Starbucks")
        st.markdown('<p class="neutral-color" style="font-size: 28px;">95.34%</p>', unsafe_allow_html=True)
        st.markdown("**F1-Score (Gradient Boosting)**")
        st.markdown("Accuracy: 95.34% | Precision: 95.38%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown("##### ☕ Kopi Kenangan")
        st.markdown('<p class="positive-color" style="font-size: 28px;">96.43%</p>', unsafe_allow_html=True)
        st.markdown("**F1-Score (SVM)**")
        st.markdown("Accuracy: 96.43% | Precision: 96.44%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Overall Statistics
    st.markdown("---")
    st.markdown("#### 📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", "5,250")
    with col2:
        st.metric("Brands", "3")
    with col3:
        st.metric("Models Tested", "6")
    with col4:
        st.metric("Best Model F1", "97.74%")
    
    # Sentiment Distribution
    st.markdown("---")
    st.markdown("#### 📈 Balanced Sentiment Distribution")
    
    all_data = pd.concat([nako, starbucks, kenangan])
    sentiment_counts = all_data['sentiment'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2ecc71', '#e74c3c']
    sentiment_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Sentimen Distribution (After Balancing)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Jumlah Reviews')
    ax.set_xlabel('Sentimen')
    ax.set_xticklabels(['Positif', 'Negatif'], rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Key Insights
    st.markdown("---")
    st.markdown("#### 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        🎯 **Kopi Nako Performance**
        - Paling mudah diklasifikasi (SVM: 97.74%)
        - Sentimen customer sangat jelas terpisah
        - Ready for production deployment
        """)
    
    with col2:
        st.info("""
        🎯 **Data Balance Success**
        - All 3 brands perfectly balanced (50-50)
        - Metrics reliable & unbiased
        - No trade-off antara precision & recall
        """)
    
    st.success("""
    ✅ **Status: PRODUCTION READY**
    Semua models siap untuk production deployment dengan confidence >95%
    """)

# ============================================================================
# PAGE 2: ANALISIS PER BRAND
# ============================================================================

elif page == "📊 Analisis Per Brand":
    st.title("📊 Analisis Sentimen Per Brand")
    
    # Select Brand
    brand = st.radio("Pilih Brand:", ["Kopi Nako", "Starbucks", "Kopi Kenangan"], horizontal=True)
    
    # Get data
    brand_data = {
        "Kopi Nako": nako,
        "Starbucks": starbucks,
        "Kopi Kenangan": kenangan
    }
    
    df = brand_data[brand]
    
    # Brand Overview
    st.markdown(f"### {brand}")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", len(df))
    with col2:
        positive_count = (df['sentiment'] == 'positive').sum()
        st.metric("Positive", positive_count)
    with col3:
        negative_count = (df['sentiment'] == 'negative').sum()
        st.metric("Negative", negative_count)
    with col4:
        st.metric("Balance Ratio", "50-50")
    
    # Sentiment Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax.pie([sentiment_counts['positive'], sentiment_counts['negative']], 
               labels=['Positif', 'Negatif'], 
               autopct='%1.1f%%',
               colors=colors,
               startangle=90)
        ax.set_title(f'Sentimen Distribution - {brand}', fontweight='bold')
        st.pyplot(fig)
    
    # Metrics Table
    with col2:
        brand_metrics = metrics[metrics['Brand'] == brand].sort_values('F1-Score', ascending=False)
        st.markdown("**Model Performance Rankings:**")
        
        best_model = brand_metrics.iloc[0]
        st.success(f"""
        🏆 **Best Model: {best_model['Model']}**
        
        - Accuracy: {best_model['Accuracy']}%
        - Precision: {best_model['Precision']}%
        - Recall: {best_model['Recall']}%
        - F1-Score: {best_model['F1-Score']}%
        """)
        
        st.markdown("**All Models:**")
        display_metrics = brand_metrics[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
        st.dataframe(display_metrics, hide_index=True, use_container_width=True)
    
    # Wordcloud
    st.markdown("---")
    st.markdown("### 📝 Wordcloud Analisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Positive Reviews Words")
        try:
            positive_text = ' '.join(df[df['sentiment'] == 'positive']['text'].values)
            positive_text = clean_text_for_wordcloud(positive_text)
            wordcloud_pos = WordCloud(width=500, height=400, 
                                     background_color='white',
                                     colormap='Greens',
                                     stopwords=COMBINED_STOPWORDS,
                                     min_font_size=12,
                                     max_words=60,
                                     collocations=False).generate(positive_text)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud_pos, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        except:
            st.warning("Tidak ada positive reviews")
    
    with col2:
        st.markdown("#### Negative Reviews Words")
        try:
            negative_text = ' '.join(df[df['sentiment'] == 'negative']['text'].values)
            negative_text = clean_text_for_wordcloud(negative_text)
            wordcloud_neg = WordCloud(width=500, height=400,
                                     background_color='white',
                                     colormap='Reds',
                                     stopwords=COMBINED_STOPWORDS,
                                     min_font_size=12,
                                     max_words=60,
                                     collocations=False).generate(negative_text)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud_neg, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        except:
            st.warning("Tidak ada negative reviews")
    
    # Sample Reviews
    st.markdown("---")
    st.markdown("### 💬 Sample Reviews")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Positive Reviews")
        positive_reviews = df[df['sentiment'] == 'positive'].sample(min(3, len(df[df['sentiment'] == 'positive'])))
        for idx, review in positive_reviews.iterrows():
            st.success(f"✓ {review['text'][:150]}...")
    
    with col2:
        st.markdown("#### Negative Reviews")
        negative_reviews = df[df['sentiment'] == 'negative'].sample(min(3, len(df[df['sentiment'] == 'negative'])))
        for idx, review in negative_reviews.iterrows():
            st.error(f"✗ {review['text'][:150]}...")

# ============================================================================
# PAGE 3: PREDIKSI SENTIMENT
# ============================================================================

elif page == "🔮 Prediksi Sentiment":
    st.title("🔮 Prediksi Sentiment Review")
    st.markdown("### Coba Model Sentiment Analysis Secara Real-time")
    
    # Load models
    st.info("Loading models... (ini loading pertama kali saja)")
    models, vectorizers = train_models()
    
    # Input
    st.markdown("---")
    st.markdown("#### Masukkan Review Anda:")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_review = st.text_area("Tulis review tentang kopi di sini:", 
                                   placeholder="Contoh: Kopinya enak banget, nyaman untuk nongkrong...",
                                   height=100)
    
    with col2:
        brand = st.selectbox("Pilih Brand:", ["Kopi Nako", "Starbucks", "Kopi Kenangan"])
    
    # Predict
    if st.button("🔮 Prediksi Sentiment", use_container_width=True):
        if user_review.strip():
            try:
                # Preprocess
                processed_text = user_review.lower()
                processed_text = ''.join(c for c in processed_text if c.isalnum() or c.isspace())
                
                # Vectorize
                X_input = vectorizers[brand].transform([processed_text])
                
                # Predict
                prediction = models[brand].predict(X_input)[0]
                confidence = models[brand].decision_function(X_input)[0]
                
                # Display Result
                st.markdown("---")
                st.markdown("#### 📊 Hasil Prediksi")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 'positive':
                        st.success(f"### ✅ POSITIF")
                        st.markdown(f"**Brand:** {brand}")
                    else:
                        st.error(f"### ❌ NEGATIF")
                        st.markdown(f"**Brand:** {brand}")
                
                with col2:
                    confidence_pct = abs(confidence) * 100
                    st.metric("Confidence Score", f"{min(confidence_pct, 100):.1f}%")
                
                # Additional Info
                st.markdown("---")
                st.info(f"""
                **Review yang dianalisis:**
                > {user_review}
                
                **Hasil Analisis:**
                - Sentimen: {prediction.upper()}
                - Confidence: {min(confidence_pct, 100):.1f}%
                - Model yang digunakan: {'SVM' if brand in ['Kopi Nako', 'Kopi Kenangan'] else 'Gradient Boosting'}
                """)
                
            except Exception as e:
                st.error(f"Error dalam prediksi: {str(e)}")
        else:
            st.warning("Masukkan review terlebih dahulu!")

# ============================================================================
# PAGE 4: PERBANDINGAN MODEL
# ============================================================================

elif page == "📈 Perbandingan Model":
    st.title("📈 Perbandingan Model Performance")
    
    st.markdown("""
    ---
    #### 📊 Metrics Explanation
    
    - **Accuracy**: Persentase prediksi yang benar dari total prediksi
    - **Precision**: Dari prediksi positif, berapa persen yang benar-benar positif
    - **Recall**: Dari semua positif aktual, berapa persen yang terdeteksi
    - **F1-Score**: Harmonic mean precision & recall (metrik terpenting)
    """)
    
    st.markdown("---")
    st.markdown("#### 📑 Tabel Lengkap Semua Metrics")
    
    # Metrics table
    st.dataframe(
        metrics.sort_values(['Brand', 'F1-Score'], ascending=[True, False]),
        hide_index=True,
        use_container_width=True
    )
    
    # Visualizations
    st.markdown("---")
    st.markdown("#### 📊 Visualisasi Perbandingan")
    
    # F1-Score by Brand
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**F1-Score Ranking**")
        fig, ax = plt.subplots(figsize=(8, 5))
        best_per_brand = metrics.sort_values('F1-Score', ascending=False).drop_duplicates('Brand')
        brands_list = best_per_brand['Brand'].values
        f1_scores = best_per_brand['F1-Score'].values
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        bars = ax.barh(brands_list, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
        
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            ax.text(score - 2, i, f'{score:.2f}%', va='center', ha='right', 
                   fontweight='bold', color='white')
        
        ax.set_xlim([90, 100])
        ax.set_xlabel('F1-Score (%)')
        ax.set_title('Best Model F1-Score per Brand', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Average Performance by Model**")
        fig, ax = plt.subplots(figsize=(8, 5))
        avg_by_model = metrics.groupby('Model')['F1-Score'].mean().sort_values(ascending=True)
        
        colors_model = plt.cm.viridis(np.linspace(0, 1, len(avg_by_model)))
        bars = ax.barh(avg_by_model.index, avg_by_model.values, color=colors_model, edgecolor='black', linewidth=1.5)
        
        for bar, score in zip(bars, avg_by_model.values):
            ax.text(score - 1, bar.get_y() + bar.get_height()/2, f'{score:.2f}%', 
                   va='center', ha='right', fontweight='bold')
        
        ax.set_xlim([85, 96])
        ax.set_xlabel('Average F1-Score (%)')
        ax.set_title('Average Performance Across All Brands', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Model comparison per brand
    st.markdown("---")
    st.markdown("**Model Comparison Per Brand**")
    
    for brand in ['Kopi Nako', 'Starbucks', 'Kopi Kenangan']:
        brand_data = metrics[metrics['Brand'] == brand].sort_values('F1-Score', ascending=False)
        
        col = st.columns(1)[0]
        with col:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            x = np.arange(len(brand_data))
            width = 0.2
            
            ax.bar(x - 1.5*width, brand_data['Accuracy'], width, label='Accuracy', alpha=0.8)
            ax.bar(x - 0.5*width, brand_data['Precision'], width, label='Precision', alpha=0.8)
            ax.bar(x + 0.5*width, brand_data['Recall'], width, label='Recall', alpha=0.8)
            ax.bar(x + 1.5*width, brand_data['F1-Score'], width, label='F1-Score', alpha=0.8, edgecolor='black', linewidth=2)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Score (%)')
            ax.set_title(f'{brand} - Model Comparison', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(brand_data['Model'], rotation=45, ha='right')
            ax.legend()
            ax.set_ylim([70, 102])
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Summary
    st.markdown("---")
    st.markdown("#### 🏆 Summary & Recommendation")
    
    best_overall = metrics.loc[metrics['F1-Score'].idxmax()]
    
    st.success(f"""
    **🏆 Best Model Overall: {best_overall['Model']}**
    
    - Brand: {best_overall['Brand']}
    - F1-Score: {best_overall['F1-Score']}%
    - Accuracy: {best_overall['Accuracy']}%
    
    **Rekomendasi untuk Production:**
    1. SVM → Deployment untuk Kopi Nako & Kopi Kenangan
    2. Gradient Boosting → Deployment untuk Starbucks
    3. Monitor F1-Score > 92% untuk production quality
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>☕ Coffee Sentiment Analysis Dashboard | Powered by Streamlit</p>
    <p>Data: 5,250 balanced reviews | Models: 6 ML algorithms | Status: ✅ Production Ready</p>
</div>
""", unsafe_allow_html=True)
