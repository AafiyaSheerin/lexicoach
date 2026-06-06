# 🎓 LexiCoach — AI Writing Improvement System

> **Single highest-impact academic vocabulary coaching powered by a custom-trained MLP neural network built from scratch in NumPy.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Model Accuracy](https://img.shields.io/badge/MLP%20Accuracy-98.08%25-brightgreen)](./04_mlp.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🧠 What is LexiCoach?

LexiCoach is an AI-powered writing coach that identifies informal vocabulary in student essays and suggests academic alternatives — using a **Multi-Layer Perceptron trained entirely from scratch** in pure NumPy (no TensorFlow, no PyTorch).

Instead of overwhelming students with every possible correction (like Grammarly's "flood model"), LexiCoach applies **Cognitive Load Theory**: it surfaces the **single highest-impact word replacement** per session, improving retention and focused learning.

---

## ✨ Features

- 🔍 **Word-level informality scoring** — Every word scored by a custom MLP (98.08% accuracy)
- 🎯 **All-words academic upgrade** — Ranked list of replacements for every informal word
- ✅ **Fully optimized passage** — One-click fully rewritten academic version
- 🔒 **Google OAuth login** — Real browser-based Sign-in with Google
- 📊 **Interactive dashboard** — Beautiful Streamlit UI with color-coded annotations
- 🧪 **No third-party ML** — Pure NumPy backpropagation, Adam optimizer, BatchNorm, Dropout

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/lexicoach.git
cd lexicoach
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK data
```python
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

### 4. Train the model (or use the included `model.pkl`)
```bash
python 04_mlp.py
```

### 5. Launch the app
```bash
streamlit run dashboard/12_app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
lexicoach/
├── dashboard/
│   └── 12_app.py              # Main Streamlit application
├── data/
│   ├── 01_eda.py              # Exploratory data analysis
│   └── 02_feature_extractor.py# Feature engineering pipeline
├── experiments/
│   ├── 06_optimizer_comparison.py
│   ├── 08_relu_vs_sigmoid.py
│   ├── 09_batchnorm_study.py
│   ├── 10_dropout_study.py
│   └── 11_depth_study.py
├── training/
│   └── ...                    # Training scripts
├── results/
│   └── plots/                 # Training convergence charts
├── 03_perceptron.py           # Linear baseline classifier
├── 04_mlp.py                  # Primary MLP model (NumPy)
├── 05_backprop_manual.py      # Manual backpropagation verification
├── model.pkl                  # Trained model weights
├── requirements.txt
└── README.md
```

---

## 🧪 Model Performance

| Metric       | Score     |
|--------------|-----------|
| **Accuracy** | **98.08%** |
| Precision    | 96.99%    |
| Recall       | 99.23%    |
| F1 Score     | 98.10%    |

**Architecture:** `Input(10) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1, Sigmoid)`  
**Trained on:** 22,571 samples from ASAP + IELTS Kaggle essay datasets  
**Optimizer:** Adam | **Regularization:** L2 + BatchNorm + Dropout(0.2)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | Pure NumPy (from scratch) |
| Web App | Streamlit |
| NLP | NLTK WordNet, wordfreq |
| Auth | Google OAuth 2.0 |
| Language | Python 3.10 |

---

## ☁️ Deploy on Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to: `dashboard/12_app.py`
5. Click **Deploy**

> **Note:** For Google OAuth, add your `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in Streamlit Cloud's **Secrets** panel.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👩‍💻 Author

**Aafiya Sheerin** — LexiCoach Final Year Project  
Built with ❤️ using pure NumPy and Streamlit
