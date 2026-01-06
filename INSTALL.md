# Installation Guide

## Quick Start (5 minutes)

### Prerequisites
- Python 3.11 or higher installed
- Git (optional, for cloning)
- 4GB RAM minimum
- 500MB disk space

### Installation Steps

#### 1. Navigate to Project Directory
```bash
cd d:\skripsi angel
```

#### 2. Create Virtual Environment
```bash
python -m venv venv
```

#### 3. Activate Virtual Environment

**Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

#### 4. Install Dependencies
```bash
pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org -r requirements.txt
```

OR install manually:
```bash
pip install pandas==2.3.3 numpy==2.4.0 matplotlib==3.10.8 seaborn==0.13.2 scikit-learn==1.8.0 nltk==3.9.2 openpyxl==3.1.5
```

#### 5. Verify Installation
```bash
python -c "import pandas, sklearn, nltk; print('✓ All packages installed successfully!')"
```

#### 6. Run the Analysis
```bash
python sentiment_analysis.py
```

---

## Detailed Installation

### Step 1: Check Python Version
```bash
python --version
```
Output should be: `Python 3.11.x` or higher

### Step 2: Verify pip is Working
```bash
pip --version
```

### Step 3: Create Virtual Environment
```bash
python -m venv venv
```
This creates a folder `venv/` with isolated Python environment

### Step 4: Activate Virtual Environment

**Important**: Always activate venv before working with the project!

Windows (PowerShell):
```bash
# You should see (venv) in your prompt after this
.\venv\Scripts\Activate.ps1
```

If you get execution policy error:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try Activate.ps1 again
```

### Step 5: Upgrade pip (Optional but Recommended)
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 6: Install Requirements
```bash
pip install -r requirements.txt
```

Or install specific packages:
```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install nltk
pip install openpyxl
```

### Step 7: Verify All Packages
```bash
pip list
```

You should see all packages from requirements.txt listed

### Step 8: Download NLTK Data (if needed)
```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

## Running the Project

### Execute Analysis
```bash
python sentiment_analysis.py
```

### Expected Output
```
================================================================================
SENTIMENT ANALYSIS MACHINE LEARNING PROJECT
================================================================================

[1] LOADING DATA...
[2] DATA CLEANING AND PREPROCESSING...
[3] TEXT PREPROCESSING...
[4] FEATURE ENGINEERING (TF-IDF)...
[5] TRAINING MACHINE LEARNING MODELS...
[6] MODEL PERFORMANCE COMPARISON
[7] DETAILED EVALUATION FOR TOP 3 MODELS
[8] GENERATING VISUALIZATIONS...
[9] FEATURE IMPORTANCE ANALYSIS

✓ Results saved to 'model_performance_results.csv'
```

### Generated Files
After successful execution, you'll have:
- ✓ model_performance_results.csv
- ✓ model_comparison.png
- ✓ roc_curve.png
- ✓ sentiment_distribution.png
- ✓ feature_importance_rf.png
- ✓ feature_importance_lr.png

---

## Troubleshooting

### Problem: "Python command not found"
**Solution**: 
- Windows: Use full path like `C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe`
- Or add Python to PATH environment variables

### Problem: "ModuleNotFoundError: No module named 'pandas'"
**Solution**:
```bash
# Make sure venv is activated
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate    # Linux/macOS

# Then install again
pip install pandas
```

### Problem: "Execution Policy" error on PowerShell
**Solution**:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problem: Slow installation due to network
**Solution**: Use the trusted hosts flag (already in requirements)
```bash
pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Problem: "Permission denied" on Linux/macOS
**Solution**:
```bash
sudo python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: NLTK stopwords not found during execution
**Solution**:
```python
python << EOF
import nltk
nltk.download('stopwords')
EOF
```

### Problem: Script runs but generates empty PNG files
**Solution**: 
This is normal - the script uses non-interactive backend. PNG files are saved to disk automatically.

---

## Uninstalling / Cleaning Up

### Deactivate Virtual Environment
```bash
deactivate
```

### Remove Virtual Environment
```bash
# Windows
rmdir /s venv

# Linux/macOS
rm -rf venv
```

### Remove Generated Files
```bash
# Windows
del model_*.csv model_*.png roc_curve.png sentiment_*.png feature_*.png

# Linux/macOS
rm -f model_*.csv model_*.png roc_curve.png sentiment_*.png feature_*.png
```

---

## Advanced Setup

### Using Anaconda (Alternative)
```bash
# Install Anaconda from https://www.anaconda.com/

# Create environment
conda create --name sentiment-analysis python=3.11

# Activate
conda activate sentiment-analysis

# Install from requirements
pip install -r requirements.txt
```

### Docker Setup (Optional)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "sentiment_analysis.py"]
```

Build and run:
```bash
docker build -t sentiment-analysis .
docker run sentiment-analysis
```

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.11 | 3.11+ |
| RAM | 4GB | 8GB |
| Disk | 500MB | 1GB |
| CPU | Dual-core | Quad-core |
| OS | Win7+/Linux/Mac | Win10+/Linux/Mac |

---

## Performance Tips

1. **Close unnecessary applications** - frees up RAM
2. **Use SSD** - faster file I/O
3. **Increase Python buffer** - set environment variable:
   ```bash
   set PYTHONUNBUFFERED=1  # Windows
   export PYTHONUNBUFFERED=1  # Linux/macOS
   ```

---

## Verification Checklist

After installation, verify everything works:

- [ ] Python 3.11+ installed
- [ ] venv created successfully
- [ ] venv activated (see (venv) in prompt)
- [ ] All packages installed (pip list shows all)
- [ ] CSV files present (kopinako_main_analysis.csv, starbucks_detailed_reviews.csv)
- [ ] Script runs without errors (python sentiment_analysis.py)
- [ ] Output files generated (PNG + CSV)

---

## Getting Help

1. **Check README.md** for detailed information
2. **Review error messages** carefully
3. **Check output logs** for specific issues
4. **Verify dataset files** exist in correct location
5. **Try clean install** - remove venv and reinstall from scratch

---

**Installation Status**: ✅ Ready to use!

Last updated: January 2026
