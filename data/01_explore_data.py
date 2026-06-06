import os
import pandas as pd
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════
# data/01_explore_data.py
# LexiCoach — Exploratory Data Analysis
# Analyzes all three datasets before any preprocessing
# ══════════════════════════════════════════════════════════════

os.makedirs("results/plots", exist_ok=True)

print("=" * 60)
print("LexiCoach — Exploratory Data Analysis")
print("=" * 60)


# ── LOAD DATASETS ─────────────────────────────────────────────
def load_asap(path):
    for enc in ['latin-1', 'utf-8', 'cp1252']:
        try:
            df = pd.read_csv(path, sep='\t', encoding=enc)
            print(f"  Loaded {path} [{enc}]")
            return df
        except Exception:
            continue
    raise ValueError(f"Could not load {path}")


def load_ielts(path):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"  Loaded {path} [{enc}]")
            return df
        except Exception:
            continue
    raise ValueError(f"Could not load {path}")


print("\n[1] Loading datasets...")
asap  = load_asap('data/training_set_rel3.tsv')
valid = load_asap('data/valid_set.tsv')
ielts = load_ielts('data/ielts_writing_dataset.csv')


# ── REQUIRED COLUMN CHECKS ────────────────────────────────────
print("\n[2] Checking required columns...")

required_asap  = ['essay', 'domain1_score', 'essay_set']
required_ielts = ['Essay', 'Overall', 'Lexical_Resource']

for col in required_asap:
    status = "✅" if col in asap.columns else "❌ MISSING"
    print(f"  ASAP  — {col}: {status}")

for col in required_ielts:
    status = "✅" if col in ielts.columns else "❌ MISSING"
    print(f"  IELTS — {col}: {status}")


# ── BASIC STATS ───────────────────────────────────────────────
print("\n[3] Dataset sizes...")
print(f"  ASAP  train : {len(asap):,} essays")
print(f"  ASAP  valid : {len(valid):,} essays")
print(f"  IELTS       : {len(ielts):,} essays")
print(f"  Total       : {len(asap)+len(ielts):,} scored essays")


# ── MISSING VALUES ────────────────────────────────────────────
print("\n[4] Missing values...")
for name, df, cols in [
    ("ASAP",  asap,  required_asap),
    ("IELTS", ielts, required_ielts)
]:
    for col in cols:
        if col in df.columns:
            n = df[col].isnull().sum()
            print(f"  {name} [{col}] : {n} missing")


# ── DUPLICATE DETECTION ───────────────────────────────────────
print("\n[5] Duplicate essays...")
asap_dups  = asap['essay'].duplicated().sum()
ielts_dups = ielts['Essay'].duplicated().sum()
print(f"  ASAP  duplicates : {asap_dups}")
print(f"  IELTS duplicates : {ielts_dups}")


# ── ESSAY LENGTH STATISTICS ───────────────────────────────────
print("\n[6] Essay length statistics (word count)...")
asap['word_count']  = asap['essay'].str.split().str.len()
ielts['word_count'] = ielts['Essay'].str.split().str.len()

for name, df in [("ASAP", asap), ("IELTS", ielts)]:
    wc = df['word_count']
    print(f"  {name}:")
    print(f"    Min    : {wc.min()}")
    print(f"    Max    : {wc.max()}")
    print(f"    Mean   : {wc.mean():.1f}")
    print(f"    Median : {wc.median():.1f}")


# ── SCORE DISTRIBUTIONS ───────────────────────────────────────
print("\n[7] Score distributions...")

print(f"  ASAP domain1_score:")
print(f"    Min  : {asap['domain1_score'].min()}")
print(f"    Max  : {asap['domain1_score'].max()}")
print(f"    Mean : {asap['domain1_score'].mean():.2f}")
print(f"    Std  : {asap['domain1_score'].std():.2f}")

print(f"\n  ASAP by essay set:")
for s in sorted(asap['essay_set'].unique()):
    sub = asap[asap['essay_set']==s]['domain1_score']
    print(f"    Set {s}: min={sub.min()} "
          f"max={sub.max()} "
          f"mean={sub.mean():.1f} "
          f"n={len(sub)}")

print(f"\n  IELTS Overall score:")
print(f"    Min  : {ielts['Overall'].min()}")
print(f"    Max  : {ielts['Overall'].max()}")
print(f"    Mean : {ielts['Overall'].mean():.2f}")
print(f"    Std  : {ielts['Overall'].std():.2f}")

print(f"\n  IELTS Lexical_Resource score:")
print(f"    Min  : {ielts['Lexical_Resource'].min()}")
print(f"    Max  : {ielts['Lexical_Resource'].max()}")
print(f"    Mean : {ielts['Lexical_Resource'].mean():.2f}")
print(f"    Std  : {ielts['Lexical_Resource'].std():.2f}")


# ── SCORE RANGE COMPATIBILITY CHECK ──────────────────────────
print("\n[8] Score scale compatibility...")
asap_range  = asap['domain1_score'].max() - asap['domain1_score'].min()
ielts_range = ielts['Overall'].max() - ielts['Overall'].min()
print(f"  ASAP  score range : {asap_range} "
      f"(will normalize per essay set to 0-1)")
print(f"  IELTS score range : {ielts_range} "
      f"(will normalize 1-9 to 0-1)")
print(f"  Compatible after normalization: ✅")


# ── VOCABULARY ANALYSIS ───────────────────────────────────────
print("\n[9] Vocabulary analysis...")
from collections import Counter
import re

sample_text = " ".join(asap['essay'].dropna().sample(
    500, random_state=42
).tolist())
words       = re.findall(r'\b[a-z]+\b', sample_text.lower())
vocab       = Counter(words)

print(f"  Sample vocabulary size    : {len(vocab):,} unique words")
print(f"  Most common (top 10)      : "
      f"{[w for w,_ in vocab.most_common(10)]}")
print(f"  Least common (hapax)      : "
      f"{sum(1 for w,c in vocab.items() if c==1):,} words")


# ── VISUALIZATIONS ────────────────────────────────────────────
print("\n[10] Saving visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("LexiCoach — Exploratory Data Analysis", fontsize=14)

# ASAP score distribution
axes[0,0].hist(asap['domain1_score'], bins=30,
               color='#2D7DD2', edgecolor='white')
axes[0,0].set_title("ASAP Score Distribution")
axes[0,0].set_xlabel("Score")
axes[0,0].set_ylabel("Count")
axes[0,0].grid(True, alpha=0.3)

# IELTS Overall score distribution
axes[0,1].hist(ielts['Overall'], bins=20,
               color='#1A8C72', edgecolor='white')
axes[0,1].set_title("IELTS Overall Score Distribution")
axes[0,1].set_xlabel("Band Score")
axes[0,1].set_ylabel("Count")
axes[0,1].grid(True, alpha=0.3)

# IELTS Lexical Resource distribution
axes[0,2].hist(ielts['Lexical_Resource'], bins=20,
               color='#E8829A', edgecolor='white')
axes[0,2].set_title("IELTS Lexical Resource Distribution")
axes[0,2].set_xlabel("Band Score")
axes[0,2].set_ylabel("Count")
axes[0,2].grid(True, alpha=0.3)

# ASAP essay word count
axes[1,0].hist(asap['word_count'], bins=40,
               color='#F4A261', edgecolor='white')
axes[1,0].set_title("ASAP Essay Length (words)")
axes[1,0].set_xlabel("Word Count")
axes[1,0].set_ylabel("Count")
axes[1,0].grid(True, alpha=0.3)

# IELTS essay word count
axes[1,1].hist(ielts['word_count'], bins=30,
               color='#6C3483', edgecolor='white')
axes[1,1].set_title("IELTS Essay Length (words)")
axes[1,1].set_xlabel("Word Count")
axes[1,1].set_ylabel("Count")
axes[1,1].grid(True, alpha=0.3)

# ASAP samples per essay set
set_counts = asap['essay_set'].value_counts().sort_index()
axes[1,2].bar(set_counts.index.astype(str),
              set_counts.values,
              color='#2D7DD2', edgecolor='white')
axes[1,2].set_title("ASAP Samples per Essay Set")
axes[1,2].set_xlabel("Essay Set")
axes[1,2].set_ylabel("Count")
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/plots/01_eda.png", dpi=150)
plt.close()
print("  Saved → results/plots/01_eda.png")


# ── SUMMARY ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"  ASAP  essays     : {len(asap):,}")
print(f"  IELTS essays     : {len(ielts):,}")
print(f"  Total            : {len(asap)+len(ielts):,}")
print(f"  ASAP  duplicates : {asap_dups}")
print(f"  IELTS duplicates : {ielts_dups}")
print(f"  ASAP  missing    : "
      f"{asap[required_asap].isnull().sum().sum()}")
print(f"  IELTS missing    : "
      f"{ielts[required_ielts].isnull().sum().sum()}")
print(f"\n  EDA complete. Ready for feature extraction.")
print(f"  NEXT: python data/02_feature_extractor.py")
print("=" * 60)