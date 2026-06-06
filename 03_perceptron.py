import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 03_perceptron.py
# LexiCoach — Linear Baseline Classifier
# Establishes performance floor before deep network pipeline
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("LexiCoach — Linear Baseline Classifier")
print("=" * 60)

# ── LOAD DATA ─────────────────────────────────────────────────
with open('data/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

assert 'X_train' in data, "Missing X_train in processed_data.pkl"
assert 'y_train' in data, "Missing y_train in processed_data.pkl"

X_train_full = data['X_train']
y_train_full = data['y_train']
X_test       = data['X_test']
y_test       = data['y_test']
feat_names   = data['feature_names']

print(f"\nDataset:")
print(f"  Train samples : {len(X_train_full):,}")
print(f"  Test samples  : {len(X_test):,}")
print(f"  Features      : {X_train_full.shape[1]}")
print(f"  Feature names : {feat_names}")
print(f"  Label 0       : keep (academic vocabulary)")
print(f"  Label 1       : replace (informal vocabulary)")

assert not np.isnan(X_train_full).any(), "NaN in X_train"
assert not np.isnan(X_test).any(),       "NaN in X_test"
print(f"  NaN check     : passed")

# ── VALIDATION SPLIT ──────────────────────────────────────────
val_size = int(0.15 * len(X_train_full))
X_val    = X_train_full[:val_size]
y_val    = y_train_full[:val_size]
X_train  = X_train_full[val_size:]
y_train  = y_train_full[val_size:]

print(f"\nSplits:")
print(f"  Train : {len(X_train):,}")
print(f"  Val   : {len(X_val):,}")
print(f"  Test  : {len(X_test):,}")


# ── PERCEPTRON ────────────────────────────────────────────────
class Perceptron:
    """
    Single-layer linear classifier.
    Trained with the perceptron learning rule.
    Best weights saved by validation F1.
    Serves as performance baseline before MLP.
    """

    def __init__(self, n_inputs, lr=0.01):
        np.random.seed(42)
        self.weights  = np.random.randn(n_inputs) * 0.01
        self.bias     = 0.0
        self.lr       = lr
        self.losses   = []
        self.val_accs = []
        self.val_f1s  = []

    def predict(self, X):
        return (X @ self.weights + self.bias >= 0).astype(float)

    def compute_metrics(self, X, y):
        preds = self.predict(X)
        tp    = int(((preds==1) & (y==1)).sum())
        tn    = int(((preds==0) & (y==0)).sum())
        fp    = int(((preds==1) & (y==0)).sum())
        fn    = int(((preds==0) & (y==1)).sum())
        acc   = np.mean(preds == y) * 100
        prec  = tp / (tp + fp + 1e-8) * 100
        rec   = tp / (tp + fn + 1e-8) * 100
        f1    = 2 * prec * rec / (prec + rec + 1e-8)
        return acc, prec, rec, f1, tp, tn, fp, fn

    def train(self, X_tr, y_tr, X_val, y_val, epochs=30):
        print(f"\nTraining linear classifier...")
        print(f"  Input features : {X_tr.shape[1]}")
        print(f"  Learning rate  : {self.lr}")
        print(f"  Epochs         : {epochs}")
        print(f"  Train samples  : {len(X_tr):,}")
        print("-" * 50)

        best_f1      = 0.0
        best_weights = self.weights.copy()
        best_bias    = self.bias

        for epoch in range(epochs):
            idx       = np.random.permutation(len(X_tr))
            X_s, y_s  = X_tr[idx], y_tr[idx]
            total_err = 0

            for xi, yi in zip(X_s, y_s):
                z         = xi @ self.weights + self.bias
                y_hat     = 1.0 if z >= 0 else 0.0
                error     = yi - y_hat
                self.weights += self.lr * error * xi
                self.bias    += self.lr * error
                total_err    += abs(error)

            avg_loss              = total_err / len(X_tr)
            val_acc,_,_,val_f1,*_ = self.compute_metrics(
                X_val, y_val
            )

            self.losses.append(avg_loss)
            self.val_accs.append(val_acc)
            self.val_f1s.append(val_f1)

            # Save best weights by validation F1
            if val_f1 > best_f1:
                best_f1      = val_f1
                best_weights = self.weights.copy()
                best_bias    = self.bias

            if epoch % 5 == 0:
                print(f"  Epoch {epoch:3d} | "
                      f"Train Err: {avg_loss:.4f} | "
                      f"Val Acc: {val_acc:.2f}% | "
                      f"Val F1: {val_f1:.2f}%")

        # Restore best weights before final evaluation
        self.weights = best_weights
        self.bias    = best_bias
        print(f"\n  Best Val F1   : {best_f1:.2f}%")
        print(f"  Weights restored to best checkpoint")


# ── TRAIN ─────────────────────────────────────────────────────
p = Perceptron(n_inputs=X_train.shape[1], lr=0.01)
p.train(X_train, y_train, X_val, y_val, epochs=30)

# ── FINAL EVALUATION ON TEST SET ──────────────────────────────
acc, prec, rec, f1, tp, tn, fp, fn = p.compute_metrics(
    X_test, y_test
)

print(f"\n{'='*60}")
print(f"BASELINE RESULTS")
print(f"{'='*60}")
print(f"  Accuracy          : {acc:.2f}%")
print(f"  Precision         : {prec:.2f}%")
print(f"  Recall            : {rec:.2f}%")
print(f"  F1 Score          : {f1:.2f}%")
print(f"  True Positives    : {tp:,}")
print(f"  True Negatives    : {tn:,}")
print(f"  False Positives   : {fp:,}")
print(f"  False Negatives   : {fn:,}")
print(f"\n  Interpretation:")
print(f"  A linear decision boundary in 4D feature space")
print(f"  achieves {acc:.1f}% accuracy and {f1:.1f}% F1.")
print(f"  This is the performance floor the MLP must exceed.")

# ── SAVE RESULTS ──────────────────────────────────────────────
with open('results/perceptron_results.pkl', 'wb') as f:
    pickle.dump({
        'weights':   p.weights,
        'bias':      p.bias,
        'accuracy':  acc,
        'f1':        f1,
        'precision': prec,
        'recall':    rec,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }, f)
print(f"\n  Results saved → results/perceptron_results.pkl")

# ── PLOT ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("LexiCoach — Linear Baseline", fontsize=13)

axes[0].plot(p.losses, color='#6C3483', linewidth=2)
axes[0].set_title("Training Error per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Avg Error")
axes[0].grid(True, alpha=0.3)

axes[1].plot(p.val_accs, color='#1A5276', linewidth=2)
axes[1].axhline(y=50, color='gray', linestyle='--',
                alpha=0.5, label='Random baseline 50%')
axes[1].set_title("Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy %")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(p.val_f1s, color='#1A8C72', linewidth=2)
axes[2].axhline(y=f1, color='red', linestyle='--',
                alpha=0.5, label=f'Test F1 {f1:.1f}%')
axes[2].set_title("Validation F1 Score")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("F1 %")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/plots/03_perceptron.png", dpi=150)
plt.close()
print(f"  Plot saved → results/plots/03_perceptron.png")
print(f"{'='*60}")
print(f"NEXT: python 04_mlp.py")
print(f"{'='*60}")