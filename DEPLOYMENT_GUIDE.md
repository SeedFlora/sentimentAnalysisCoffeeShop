# 🚀 DEPLOYMENT GUIDE - STREAMLIT LIVE

## 📋 QUICK START DEPLOYMENT (15 menit)

### STEP 1: GitHub Setup (5 menit)

#### 1.1 Buat GitHub Account
- Pergi ke https://github.com
- Sign up dengan email
- Verify email Anda

#### 1.2 Buat Repository
- Login ke GitHub
- Klik `+` → `New repository`
- **Name**: `sentiment-analysis-kopi`
- **Visibility**: PUBLIC ✅
- **Initialize with**: Add .gitignore (Python)
- Click `Create repository`
- **COPY HTTPS URL repo Anda**

---

### STEP 2: Push Code ke GitHub (5 menit)

Buka PowerShell di `D:\skripsi angel`:

```powershell
cd "D:\skripsi angel"

# Setup git (first time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@gmail.com"

# Initialize & push
git init
git add .
git commit -m "Initial: Sentiment Analysis Dashboard with Streamlit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sentiment-analysis-kopi.git
git push -u origin main
```

Jika minta password:
- **Username**: GitHub username Anda
- **Password**: Personal Access Token (buat di GitHub Settings)

---

### STEP 3: Deploy ke Streamlit Cloud (5 menit)

1. **Pergi ke** https://streamlit.io/cloud
2. **Sign up with GitHub** (autorize Streamlit)
3. **Click "New app"**
4. **Fill:**
   - **GitHub account**: Pilih GitHub Anda
   - **Repository**: `sentiment-analysis-kopi`
   - **Branch**: `main`
   - **File path**: `streamlit_app.py`
5. **Click "Deploy"**
6. **Wait 2-3 menit...**
7. **🎉 LIVE! Your app is at:**
   ```
   https://sentiment-analysis-kopi.streamlit.app
   ```

---

## 📁 Files yang Harus Ada di GitHub
---

### Option 2: Deploy di Server Pribadi/VPS

#### Step 1: Install di Server
```bash
# SSH ke server
ssh user@your-server.com

# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip

# Clone repository
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-kopi.git
cd sentiment-analysis-kopi

# Install Python dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run streamlit_app.py --server.port 80 --server.address 0.0.0.0
```

#### Step 2: Setup Reverse Proxy (nginx)
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

#### Step 3: Setup SSL dengan Let's Encrypt
```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

---

### Option 3: Deploy sebagai Docker Container

#### Step 1: Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
```

#### Step 2: Build & Run
```bash
# Build
docker build -t sentiment-kopi .

# Run
docker run -p 8501:8501 sentiment-kopi
```

#### Step 3: Deploy to Docker Hub / AWS / Heroku
```bash
# Push ke Docker Hub
docker login
docker tag sentiment-kopi YOUR_USERNAME/sentiment-kopi
docker push YOUR_USERNAME/sentiment-kopi
```

---

## 📁 REQUIRED FILES

Pastikan semua files ini ada di GitHub repo:

```
sentiment-analysis-kopi/
├── streamlit_app.py          ✅ (MAIN APP - REQUIRED)
├── requirements.txt          ✅ (DEPENDENCIES - REQUIRED)
├── .gitignore               ✅ (GIT IGNORE)
│
├── 📊 DATA FILES
├── kopi_nako_balanced.csv   ✅
├── starbucks_balanced.csv   ✅
├── kopi_kenangan_balanced.csv ✅
├── per_brand_evaluation_results.csv
│
├── 🤖 MODEL FILES (optional, bisa di Git LFS)
├── svm_model_kopi_nako.pkl
├── svm_model_kopi_kenangan.pkl
├── gb_model_starbucks.pkl
├── tfidf_vectorizer.pkl
│
├── 📚 DOCUMENTATION
├── README.md
├── DEPLOYMENT_GUIDE.md
├── METRICS_FINAL_SUMMARY.txt
│
└── 🎨 VISUALIZATIONS (optional)
    ├── per_brand_metrics_comparison.png
    ├── per_brand_models_comparison.png
    └── ... (other visualization files)
```

---

## ⚙️ REQUIREMENTS.TXT MUST HAVE
# SSH ke server
ssh user@server.com

# Install Python 3.11
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv

# Clone repository
git clone https://github.com/USERNAME/coffee-sentiment-analysis.git
cd coffee-sentiment-analysis

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Run Streamlit
```bash
# Option A: Development (local access only)
streamlit run app.py

# Option B: Production (accessible from web)
streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0
```

#### Step 3: Use Nginx as Reverse Proxy
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

#### Step 4: Use Systemd Service (untuk auto-start)
```ini
# File: /etc/systemd/system/streamlit.service

[Unit]
Description=Streamlit Dashboard
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/home/user/coffee-sentiment-analysis
ExecStart=/home/user/coffee-sentiment-analysis/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable service
sudo systemctl enable streamlit
sudo systemctl start streamlit
sudo systemctl status streamlit
```

---

### Option 3: Deploy di Heroku

#### Step 1: Create Heroku Account
- Signup di https://www.heroku.com

#### Step 2: Create Procfile
```
# File: Procfile
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

#### Step 3: Deploy
```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

---

## 📝 Streamlit Configuration (Optional)

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#2ecc71"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#000000"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true

[client]
showErrorDetails = true

[browser]
gatherUsageStats = false
```

---

## 🔒 Security Tips

### 1. Add .gitignore
```
# Don't push sensitive data
*.pkl
*.joblib
*.env
.streamlit/secrets.toml
```

### 2. Use Streamlit Secrets (untuk API keys, dll)
```python
# In app.py
import streamlit as st

secret_key = st.secrets["my_secret_key"]
```

Create `.streamlit/secrets.toml`:
```toml
my_secret_key = "your-secret-here"
```

### 3. HTTPS
- Streamlit Cloud: automatic HTTPS ✅
- VPS: Use Let's Encrypt (certbot)
- Heroku: automatic HTTPS ✅

---

## 🚀 Quick Start Commands

### Local Development
```bash
cd "D:\skripsi angel"
.\venv\Scripts\Activate.ps1  # Windows
streamlit run app.py
```

**Browser akan otomatis buka:** http://localhost:8501

### GitHub Deployment
```bash
git add .
git commit -m "Deploy dashboard"
git push origin main
```

Streamlit Cloud akan otomatis rebuild & deploy! 🎉

---

## 📊 Dashboard Features

✅ **Dashboard Utama**
- Project overview
- Best model performance metrics
- Key insights

✅ **Analisis Per Brand**
- Brand-specific metrics
- Sentiment distribution (pie chart)
- Model performance rankings
- **Wordcloud Analysis** (positive & negative words)
- Sample reviews

✅ **Prediksi Sentiment**
- Real-time sentiment prediction
- Interactive input
- Confidence score
- Model selection

✅ **Perbandingan Model**
- F1-Score ranking
- Performance comparison
- Model metrics table
- Visual charts

---

## 🎯 File Structure

```
coffee-sentiment-analysis/
├── app.py                              # Main dashboard app
├── requirements.txt                    # Dependencies
├── .gitignore                         # Git ignore rules
├── README.md                          # Project documentation
├── kopi_nako_balanced.csv             # Dataset 1
├── starbucks_balanced.csv             # Dataset 2
├── kopi_kenangan_balanced.csv         # Dataset 3
├── per_brand_evaluation_results.csv   # Metrics
├── Procfile                           # Heroku config (optional)
├── .streamlit/                        # Streamlit config
│   └── config.toml                   # Theme & settings
└── venv/                              # Virtual environment (don't push)
```

---

## ✅ Testing Checklist

Sebelum deploy, test:

- [ ] `streamlit run app.py` berjalan tanpa error
- [ ] Semua 4 halaman bisa diakses
- [ ] Wordcloud muncul dengan baik
- [ ] Prediksi sentiment berfungsi
- [ ] Data loaded dengan benar
- [ ] Visualisasi tampil dengan baik
- [ ] Responsive di mobile device

---

## 🆘 Troubleshooting

### Error: "No such file or directory: 'kopi_nako_balanced.csv'"
**Solution:** Ensure semua CSV files ada di same folder dengan `app.py`

### Error: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** 
```bash
pip install streamlit
# atau
.\venv\Scripts\python.exe -m pip install streamlit
```

### Slow loading di first time
**Solution:** Normal! Streamlit cache mengoptimasi loading. Refresh page akan lebih cepat.

### Wordcloud error
**Solution:** Ensure review text bukan kosong. Check di CSV file.

---

## 📈 Post-Deployment Monitoring

### Streamlit Cloud
- Dashboard built-in di https://share.streamlit.io
- Monitor usage, errors, logs

### Custom Server
```bash
# Check if running
sudo systemctl status streamlit

# View logs
sudo journalctl -u streamlit -f

# Restart
sudo systemctl restart streamlit
```

---

## 🎉 SUCCESS CHECKLIST

- [ ] Dashboard deployed dan accessible via URL
- [ ] All 4 pages working
- [ ] Wordcloud displayed
- [ ] Sentiment prediction working
- [ ] Metrics showing correctly
- [ ] Mobile responsive
- [ ] Fast loading (< 3 seconds)

---

**Status: ✅ READY FOR PRODUCTION**

Choose your deployment option dan go live! 🚀

