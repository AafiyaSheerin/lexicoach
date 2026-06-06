import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════
# 04_mlp.py
# LexiCoach — Multi-Layer Perceptron (Pure NumPy)
# ══════════════════════════════════════════════════════════════

class MLP:
    """
    Multi-Layer Perceptron from scratch in NumPy.
    Supports dynamic layers, Dropout, Batch Normalization, L2 regularization,
    gradient clipping, and SGD or Adam optimization.
    """
    def __init__(self, layer_sizes, activation='relu', use_batchnorm=False,
                 dropout_rate=0.0, l2_reg=0.001, optimizer='adam', lr=0.001,
                 beta1=0.9, beta2=0.999, epsilon=1e-8, grad_clip=5.0):
        self.layer_sizes = layer_sizes
        self.activation = activation.lower()
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.optimizer = optimizer.lower()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        
        self.W = []
        self.b = []
        self.gamma = []
        self.beta = []
        self.running_mean = []
        self.running_var = []
        
        self.initialize_parameters()
        self.reset_optimizer()
        
    def initialize_parameters(self):
        np.random.seed(42)
        L = len(self.layer_sizes)
        for i in range(L - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i+1]
            
            # He initialization for ReLU, Xavier for Sigmoid
            if self.activation == 'relu' and i < L - 2:
                scale = np.sqrt(2.0 / n_in)
            else:
                scale = np.sqrt(1.0 / n_in)
                
            self.W.append(np.random.randn(n_in, n_out) * scale)
            self.b.append(np.zeros((1, n_out)))
            
            # Initialize BatchNorm parameters for hidden layers
            if self.use_batchnorm and i < L - 2:
                self.gamma.append(np.ones((1, n_out)))
                self.beta.append(np.zeros((1, n_out)))
                self.running_mean.append(np.zeros((1, n_out)))
                self.running_var.append(np.ones((1, n_out)))
            else:
                # Placeholders to maintain list alignment
                self.gamma.append(None)
                self.beta.append(None)
                self.running_mean.append(None)
                self.running_var.append(None)

    def reset_optimizer(self):
        self.t = 0
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        
        self.mgamma = []
        self.vgamma = []
        self.mbeta = []
        self.vbeta = []
        for g, b in zip(self.gamma, self.beta):
            if g is not None:
                self.mgamma.append(np.zeros_like(g))
                self.vgamma.append(np.zeros_like(g))
                self.mbeta.append(np.zeros_like(b))
                self.vbeta.append(np.zeros_like(b))
            else:
                self.mgamma.append(None)
                self.vgamma.append(None)
                self.mbeta.append(None)
                self.vbeta.append(None)

    def forward(self, X, training=True):
        self.cache = {}
        self.cache['a0'] = X
        a = X
        L = len(self.layer_sizes)
        
        for l in range(L - 2):  # Hidden layers
            W = self.W[l]
            b = self.b[l]
            z = a @ W + b
            self.cache[f'z{l+1}'] = z
            
            # Batch Normalization step (pre-activation)
            if self.use_batchnorm:
                if training:
                    mean = np.mean(z, axis=0, keepdims=True)
                    var = np.var(z, axis=0, keepdims=True)
                    # Update running stats
                    self.running_mean[l] = 0.9 * self.running_mean[l] + 0.1 * mean
                    self.running_var[l] = 0.9 * self.running_var[l] + 0.1 * var
                    z_hat = (z - mean) / np.sqrt(var + 1e-5)
                else:
                    z_hat = (z - self.running_mean[l]) / np.sqrt(self.running_var[l] + 1e-5)
                
                z_bn = self.gamma[l] * z_hat + self.beta[l]
                self.cache[f'z_hat{l+1}'] = z_hat
                self.cache[f'z_bn{l+1}'] = z_bn
                pre_activation = z_bn
            else:
                pre_activation = z
                
            # Activation
            if self.activation == 'relu':
                a = np.maximum(0, pre_activation)
            else:
                a = 1.0 / (1.0 + np.exp(-np.clip(pre_activation, -500, 500)))
            
            # Dropout (training only)
            if training and self.dropout_rate > 0.0:
                mask = (np.random.rand(*a.shape) >= self.dropout_rate).astype(float)
                a = a * mask / (1.0 - self.dropout_rate)
                self.cache[f'mask{l+1}'] = mask
                
            self.cache[f'a{l+1}'] = a
            
        # Output layer
        W = self.W[-1]
        b = self.b[-1]
        z = a @ W + b
        self.cache[f'z{L-1}'] = z
        a_out = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        self.cache[f'a{L-1}'] = a_out
        
        return a_out

    def backward(self, X, y):
        grads = {
            'dW': [None] * len(self.W),
            'db': [None] * len(self.b),
            'dgamma': [None] * len(self.gamma),
            'dbeta': [None] * len(self.beta)
        }
        
        L = len(self.layer_sizes)
        m = X.shape[0]
        y = y.reshape(-1, 1)
        
        a_out = self.cache[f'a{L-1}']
        # Binary Cross Entropy + Sigmoid derivative dL/dz3
        dz = (a_out - y) / m
        
        a_prev = self.cache[f'a{L-2}']
        grads['dW'][-1] = a_prev.T @ dz + (self.l2_reg * self.W[-1]) / m
        grads['db'][-1] = np.sum(dz, axis=0, keepdims=True)
        
        da = dz @ self.W[-1].T
        
        # Hidden layers backward
        for l in range(L - 2, 0, -1):
            idx = l - 1
            
            # Dropout backward
            if self.dropout_rate > 0.0:
                mask = self.cache[f'mask{l}']
                da = da * mask / (1.0 - self.dropout_rate)
                
            # Activation backward
            if self.use_batchnorm:
                pre_activation = self.cache[f'z_bn{l}']
            else:
                pre_activation = self.cache[f'z{l}']
                
            if self.activation == 'relu':
                dpre = da * (pre_activation > 0).astype(float)
            else:
                a_act = self.cache[f'a{l}']
                dpre = da * a_act * (1.0 - a_act)
                
            # BatchNorm backward
            if self.use_batchnorm:
                z_hat = self.cache[f'z_hat{l}']
                grads['dgamma'][idx] = np.sum(dpre * z_hat, axis=0, keepdims=True)
                grads['dbeta'][idx] = np.sum(dpre, axis=0, keepdims=True)
                
                # Gradient through scale & shift
                dhat = dpre * self.gamma[idx]
                z_raw = self.cache[f'z{l}']
                var = np.var(z_raw, axis=0, keepdims=True)
                std = np.sqrt(var + 1e-5)
                
                # Exact BatchNorm backward formula
                dpre_bn = (1.0 / (m * std)) * (
                    m * dhat -
                    np.sum(dhat, axis=0, keepdims=True) -
                    z_hat * np.sum(dhat * z_hat, axis=0, keepdims=True)
                )
                dz_layer = dpre_bn
            else:
                dz_layer = dpre
                
            # Linear backward
            a_prev = self.cache[f'a{idx}']
            grads['dW'][idx] = a_prev.T @ dz_layer + (self.l2_reg * self.W[idx]) / m
            grads['db'][idx] = np.sum(dz_layer, axis=0, keepdims=True)
            
            da = dz_layer @ self.W[idx].T
            
        return grads

    def update_parameters(self, grads):
        self.t += 1
        for i in range(len(self.W)):
            # Gradient clipping
            dW = np.clip(grads['dW'][i], -self.grad_clip, self.grad_clip)
            db = np.clip(grads['db'][i], -self.grad_clip, self.grad_clip)
            
            if self.optimizer == 'adam':
                # Momentum W
                self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * dW
                # Velocity W
                self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (dW ** 2)
                # Bias corrected
                mW_corrected = self.mW[i] / (1.0 - self.beta1 ** self.t)
                vW_corrected = self.vW[i] / (1.0 - self.beta2 ** self.t)
                # Update
                self.W[i] -= self.lr * mW_corrected / (np.sqrt(vW_corrected) + self.epsilon)
                
                # Momentum b
                self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * db
                # Velocity b
                self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (db ** 2)
                # Bias corrected
                mb_corrected = self.mb[i] / (1.0 - self.beta1 ** self.t)
                vb_corrected = self.vb[i] / (1.0 - self.beta2 ** self.t)
                # Update
                self.b[i] -= self.lr * mb_corrected / (np.sqrt(vb_corrected) + self.epsilon)
                
                # Update BatchNorm parameters if they exist
                if self.use_batchnorm and i < len(self.W) - 1:
                    dgamma = np.clip(grads['dgamma'][i], -self.grad_clip, self.grad_clip)
                    dbeta = np.clip(grads['dbeta'][i], -self.grad_clip, self.grad_clip)
                    
                    self.mgamma[i] = self.beta1 * self.mgamma[i] + (1 - self.beta1) * dgamma
                    self.vgamma[i] = self.beta2 * self.vgamma[i] + (1 - self.beta2) * (dgamma ** 2)
                    mgamma_corrected = self.mgamma[i] / (1.0 - self.beta1 ** self.t)
                    vgamma_corrected = self.vgamma[i] / (1.0 - self.beta2 ** self.t)
                    self.gamma[i] -= self.lr * mgamma_corrected / (np.sqrt(vgamma_corrected) + self.epsilon)
                    
                    self.mbeta[i] = self.beta1 * self.mbeta[i] + (1 - self.beta1) * dbeta
                    self.vbeta[i] = self.beta2 * self.vbeta[i] + (1 - self.beta2) * (dbeta ** 2)
                    mbeta_corrected = self.mbeta[i] / (1.0 - self.beta1 ** self.t)
                    vbeta_corrected = self.vbeta[i] / (1.0 - self.beta2 ** self.t)
                    self.beta[i] -= self.lr * mbeta_corrected / (np.sqrt(vbeta_corrected) + self.epsilon)
            else:
                # Standard SGD
                self.W[i] -= self.lr * dW
                self.b[i] -= self.lr * db
                if self.use_batchnorm and i < len(self.W) - 1:
                    self.gamma[i] -= self.lr * np.clip(grads['dgamma'][i], -self.grad_clip, self.grad_clip)
                    self.beta[i] -= self.lr * np.clip(grads['dbeta'][i], -self.grad_clip, self.grad_clip)

    def compute_loss(self, y_pred, y):
        m = y.shape[0]
        y = y.reshape(-1, 1)
        # Binary Cross-Entropy loss
        loss = -1.0 / m * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
        # Add L2 Regularization loss
        l2_loss = 0.5 * self.l2_reg / m * sum(np.sum(w ** 2) for w in self.W)
        return loss + l2_loss

    def predict_proba(self, X):
        return self.forward(X, training=False)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X, y, threshold=0.5):
        preds = self.predict(X, threshold)
        y = y.reshape(-1, 1)
        
        tp = np.sum((preds == 1) & (y == 1))
        tn = np.sum((preds == 0) & (y == 0))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))
        
        acc = np.mean(preds == y) * 100
        prec = tp / (tp + fp + 1e-15) * 100
        rec = tp / (tp + fn + 1e-15) * 100
        f1 = 2 * prec * rec / (prec + rec + 1e-15)
        
        return acc, prec, rec, f1, int(tp), int(tn), int(fp), int(fn)


# ── MAIN PIPELINE FOR 04_MLP ──────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Training Primary MLP model on LexiCoach preprocessed dataset")
    print("=" * 60)
    
    # Load dataset
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        
    X_train_full = data['X_train']
    y_train_full = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    # Validation split (15%)
    val_size = int(0.15 * len(X_train_full))
    X_val = X_train_full[:val_size]
    y_val = y_train_full[:val_size]
    X_train = X_train_full[val_size:]
    y_train = y_train_full[val_size:]
    
    print(f"Train samples      : {X_train.shape[0]:,}")
    print(f"Validation samples : {X_val.shape[0]:,}")
    print(f"Test samples       : {X_test.shape[0]:,}")
    print(f"Input features     : {X_train.shape[1]}")
    
    # Initialize MLP matching target config
    # Input(10) -> Hidden 1(64) -> Hidden 2(32) -> Output(1)
    mlp = MLP(
        layer_sizes=[X_train.shape[1], 64, 32, 1],
        activation='relu',
        use_batchnorm=True,
        dropout_rate=0.2,
        l2_reg=0.001,
        optimizer='adam',
        lr=0.001
    )
    
    epochs = 100
    batch_size = 256
    patience = 15
    
    train_losses = []
    val_losses = []
    val_f1s = []
    
    best_val_loss = float('inf')
    best_weights = None
    best_biases = None
    best_gamma = None
    best_beta = None
    best_running_mean = None
    best_running_var = None
    
    patience_counter = 0
    
    print("\nTraining primary MLP classifier...")
    for epoch in range(epochs):
        # Shuffle train data
        indices = np.random.permutation(X_train.shape[0])
        X_tr_shuffled = X_train[indices]
        y_tr_shuffled = y_train[indices]
        
        epoch_losses = []
        for start in range(0, X_train.shape[0], batch_size):
            end = start + batch_size
            xb = X_tr_shuffled[start:end]
            yb = y_tr_shuffled[start:end]
            
            y_pred = mlp.forward(xb, training=True)
            grads = mlp.backward(xb, yb)
            mlp.update_parameters(grads)
            
            loss = mlp.compute_loss(y_pred, yb)
            epoch_losses.append(loss)
            
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        
        # Validation evaluation
        y_val_pred = mlp.forward(X_val, training=False)
        val_loss = mlp.compute_loss(y_val_pred, y_val)
        val_losses.append(val_loss)
        
        val_acc, _, _, val_f1, *_ = mlp.evaluate(X_val, y_val, threshold=0.5)
        val_f1s.append(val_f1)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save checkpoints
            best_weights = [w.copy() for w in mlp.W]
            best_biases = [b.copy() for b in mlp.b]
            best_gamma = [g.copy() if g is not None else None for g in mlp.gamma]
            best_beta = [b.copy() if b is not None else None for b in mlp.beta]
            best_running_mean = [rm.copy() if rm is not None else None for rm in mlp.running_mean]
            best_running_var = [rv.copy() if rv is not None else None for rv in mlp.running_var]
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.2f}%")
            
        if patience_counter >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break
            
    # Restore best checkpoint parameters
    mlp.W = best_weights
    mlp.b = best_biases
    mlp.gamma = best_gamma
    mlp.beta = best_beta
    mlp.running_mean = best_running_mean
    mlp.running_var = best_running_var
    
    # Threshold Tuning to reach max F1 / Accuracy
    best_threshold = 0.5
    best_val_acc = 0.0
    for th in np.linspace(0.1, 0.9, 81):
        v_acc, _, _, _, *_ = mlp.evaluate(X_val, y_val, threshold=th)
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_threshold = th
            
    print(f"\nOptimized Decision Threshold on Validation: {best_threshold:.3f} (Val Accuracy: {best_val_acc:.2f}%)")
    
    # Final Test Set Evaluation
    t_acc, t_prec, t_rec, t_f1, tp, tn, fp, fn = mlp.evaluate(X_test, y_test, threshold=best_threshold)
    
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS (PRIMARY MLP)")
    print("=" * 60)
    print(f"  Accuracy          : {t_acc:.2f}%")
    print(f"  Precision         : {t_prec:.2f}%")
    print(f"  Recall            : {t_rec:.2f}%")
    print(f"  F1 Score          : {t_f1:.2f}%")
    print(f"  True Positives    : {tp:,}")
    print(f"  True Negatives    : {tn:,}")
    print(f"  False Positives   : {fp:,}")
    print(f"  False Negatives   : {fn:,}")
    
    # Save the trained model checkpoint
    model_checkpoint = {
        'weights': mlp.W,
        'biases': mlp.b,
        'gamma': mlp.gamma,
        'beta': mlp.beta,
        'running_mean': mlp.running_mean,
        'running_var': mlp.running_var,
        'layer_sizes': mlp.layer_sizes,
        'activation': mlp.activation,
        'use_batchnorm': mlp.use_batchnorm,
        'threshold': best_threshold,
        'feature_names': feature_names,
        'word_freq': data['word_freq'],
        'total_freq': data['total_freq'],
        'informal_words': data['informal_words'],
        'academic_words': data['academic_words']
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_checkpoint, f)
        
    print(f"\nModel and weights saved successfully -> model.pkl")
    
    # Save training loss plots
    os.makedirs("results/plots", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss', color='#6C3483', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='#1A5276', linewidth=2)
    plt.axvline(x=len(train_losses) - patience, color='red', linestyle='--', alpha=0.5, label='Early Stopping checkpoint')
    plt.title('MLP Training Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/04_mlp_loss.png', dpi=150)
    plt.close()
    print("Plot saved -> results/plots/04_mlp_loss.png")