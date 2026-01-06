# ⚡ QUICK DEPLOYMENT CHECKLIST

## 1️⃣ SETUP GIT LOKAL (First Time Only)

```powershell
cd "D:\skripsi angel"

# Configure git
git config --global user.name "Your Full Name"
git config --global user.email "your.email@gmail.com"

# Check configuration
git config --list
```

---

## 2️⃣ BUAT GITHUB REPO

1. **Pergi ke https://github.com**
2. **Sign up / Login**
3. **Click `+` icon → New repository**
4. **Fill form:**
   - Repository name: `sentiment-analysis-kopi`
   - Description: `Coffee Sentiment Analysis with ML Models`
   - Visibility: **PUBLIC** ✅
   - Initialize: Add .gitignore (Python)
5. **Click "Create repository"**
6. **Copy HTTPS URL** (akan butuh sebentar)

---

## 3️⃣ PUSH KE GITHUB

```powershell
cd "D:\skripsi angel"

# Initialize git repo
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Sentiment Analysis Dashboard with Streamlit"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/sentiment-analysis-kopi.git

# Rename branch to main
git branch -M main

# Push
git push -u origin main
```

**Jika minta password:**
- Username: Your GitHub username
- Password: Personal Access Token (create di GitHub Settings)

**Cara buat Personal Access Token:**
1. Login ke GitHub
2. Settings → Developer settings → Personal access tokens
3. Generate new token
4. Pilih: `repo`, `workflow`, `gist`
5. Copy token → gunakan sebagai password

---

## 4️⃣ DEPLOY KE STREAMLIT CLOUD

1. **Pergi ke https://streamlit.io/cloud**
2. **Click "Sign in with GitHub"**
3. **Authorize Streamlit untuk akses GitHub Anda**
4. **Click "New app" button**
5. **Fill form:**
   ```
   GitHub account    → Pilih GitHub Anda
   Repository        → sentiment-analysis-kopi
   Branch            → main
   File path         → streamlit_app.py
   ```
6. **Click "Deploy"**
7. **Tunggu 2-3 menit...**

---

## ✅ VERIFY DEPLOYMENT

Setelah deploy berhasil:

- [ ] Streamlit Cloud dashboard tidak error
- [ ] App load dalam 30 detik
- [ ] Dashboard tab terbuka tanpa error
- [ ] Sidebar menu berfungsi
- [ ] Dropdown brand bisa dipilih
- [ ] Visualisasi muncul
- [ ] WordCloud muncul
- [ ] Prediction form berfungsi

---

## 🔄 UPDATE APP (Setelah Deployment)

Setiap kali Anda update code:

```powershell
cd "D:\skripsi angel"

# Edit files Anda
# ...

# Stage changes
git add .

# Commit
git commit -m "Update: Description of changes"

# Push
git push
```

**Streamlit auto-rerun dalam 1-2 menit!** ✅

---

## 🐛 TROUBLESHOOTING

### Error: "Unable to deploy"
✅ **Solution:**
- Push code ke GitHub dulu (Step 3)
- Pastikan repo PUBLIC (bukan private)
- Tunggu 2-3 menit, terus refresh

### Error: "ModuleNotFoundError: No module named 'xxx'"
✅ **Solution:**
- Pastikan package di `requirements.txt`
- Jalankan: `pip freeze > requirements.txt`
- Push updated requirements.txt
- Click "Rerun" di Streamlit dashboard

### Error: "FileNotFoundError: 'kopi_nako_balanced.csv'"
✅ **Solution:**
- Pastikan CSV files ada di GitHub repo
- Gunakan path relatif (bukan absolute path)
- Push files ke GitHub

### App Timeout / Crash
✅ **Solution:**
- Check Streamlit logs (click "Logs" di dashboard)
- Reduce data size (pastikan CSV < 50MB)
- Add caching decorator (`@st.cache_data`)
- Reduce visualizations complexity

### Too Many Requests / Rate Limited
✅ **Solution:**
- Streamlit Cloud Community free untuk 50 concurrent users
- Upgrade ke Pro plan jika lebih banyak
- Atau host di server pribadi

---

## 📊 YOUR LIVE APP URL

Setelah deploy, app Anda akan di:

```
https://sentiment-analysis-kopi.streamlit.app
```

(ganti dengan nama repo Anda)

---

## 🎯 NEXT STEPS

### Share App
- [ ] Copy URL app
- [ ] Share ke friends/colleagues/professor
- [ ] Embed di website (iframe)
- [ ] QR code ke URL

### Monitor Usage
- [ ] Check Streamlit Cloud dashboard
- [ ] Monitor CPU/Memory usage
- [ ] Read logs untuk errors
- [ ] Track unique visitors

### Improve Performance
- [ ] Add more caching
- [ ] Optimize visualizations
- [ ] Reduce file sizes
- [ ] Add pagination untuk large datasets

### Add Features
- [ ] Export results ke PDF
- [ ] Batch prediction upload CSV
- [ ] More visualizations
- [ ] API endpoint

---

## 📞 HELP RESOURCES

**Streamlit Docs**: https://docs.streamlit.io  
**GitHub Help**: https://docs.github.com  
**Streamlit Community**: https://discuss.streamlit.io  

---

## ✨ SUMMARY

| Step | Action | Time | Status |
|------|--------|------|--------|
| 1 | Git config | 2 min | ⏳ |
| 2 | GitHub repo | 3 min | ⏳ |
| 3 | Push code | 5 min | ⏳ |
| 4 | Deploy Streamlit | 5 min | ⏳ |
| 5 | Wait & verify | 3 min | ⏳ |
| **TOTAL** | | **18 min** | 🚀 |

---

**Good luck! Your app will be live soon! 🎉**
