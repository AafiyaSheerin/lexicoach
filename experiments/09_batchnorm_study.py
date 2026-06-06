import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# Add root directory to path to import 04_mlp.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
spec = importlib.util.spec_from_file_location("mlp", os.path.abspath(os.path.join(os.path.dirname(__file__), "../04_mlp.py")))
mlp_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp_mod)
MLP = mlp_mod.MLP

# ══════════════════════════════════════════════════════════════
# experiments/09_batchnorm_study.py
# LexiCoach — Batch Normalization Convergence Study
# ══════════════════════════════════════════════════════════════

def train_network(use_bn, X_tr, y_tr, X_val, y_val, epochs=25, lr=0.005):
    mlp = MLP(
        layer_sizes=[X_tr.shape[1], 64, 32, 1],
        activation='relu',
        use_batchnorm=use_bn,
        dropout_rate=0.0,
        l2_reg=0.001,
        optimizer='adam',
        lr=lr
    )
    
    losses = []
    val_losses = []
    
    batch_size = 256
    for epoch in range(epochs):
        indices = np.random.permutation(X_tr.shape[0])
        X_tr_sh = X_tr[indices]
        y_tr_sh = y_tr[indices]
        
        epoch_losses = []
        for start in range(0, X_tr.shape[0], batch_size):
            end = start + batch_size
            xb = X_tr_sh[start:end]
            yb = y_tr_sh[start:end]
            
            y_pred = mlp.forward(xb, training=True)
            grads = mlp.backward(xb, yb)
            mlp.update_parameters(grads)
            
            loss = mlp.compute_loss(y_pred, yb)
            epoch_losses.append(loss)
            
        losses.append(np.mean(epoch_losses))
        
        y_val_pred = mlp.forward(X_val, training=False)
        val_loss = mlp.compute_loss(y_val_pred, y_val)
        val_losses.append(val_loss)
        
    return losses, val_losses


if __name__ == '__main__':
    print("=" * 60)
    print("LexiCoach — Batch Normalization Study")
    print("=" * 60)
    
    # Load dataset
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        
    X_train_full = data['X_train']
    y_train_full = data['y_train']
    
    val_size = int(0.15 * len(X_train_full))
    X_val = X_train_full[:val_size]
    y_val = y_train_full[:val_size]
    X_train = X_train_full[val_size:]
    y_train = y_train_full[val_size:]
    
    epochs = 20
    
    # Run with high learning rate (0.01) to show stabilization effect
    lr = 0.01
    print(f"Training with BatchNorm (lr={lr})...")
    bn_train, bn_val = train_network(True, X_train, y_train, X_val, y_val, epochs, lr)
    
    print(f"Training without BatchNorm (lr={lr})...")
    nobn_train, nobn_val = train_network(False, X_train, y_train, X_val, y_val, epochs, lr)
    
    # Save the plot
    os.makedirs("results/plots", exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(nobn_train, label='Without BatchNorm (Train)', color='#E74C3C', linestyle='--', linewidth=2)
    plt.plot(nobn_val, label='Without BatchNorm (Val)', color='#C0392B', linewidth=2)
    
    plt.plot(bn_train, label='With BatchNorm (Train)', color='#2ECC71', linestyle='--', linewidth=2)
    plt.plot(bn_val, label='With BatchNorm (Val)', color='#27AE60', linewidth=2)
    
    plt.title('Batch Normalization Study: Convergence & Stability')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/09_batchnorm_study.png', dpi=150)
    plt.close()
    print("\nSaved comparison plot to -> results/plots/09_batchnorm_study.png")
