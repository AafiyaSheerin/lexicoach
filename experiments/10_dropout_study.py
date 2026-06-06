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
# experiments/10_dropout_study.py
# LexiCoach — Dropout Study (Overfitting Prevention)
# ══════════════════════════════════════════════════════════════

def train_with_dropout(rate, X_tr, y_tr, X_val, y_val, epochs=25):
    print(f"Training with Dropout Rate: {rate}...")
    mlp = MLP(
        layer_sizes=[X_tr.shape[1], 64, 32, 1],
        activation='relu',
        use_batchnorm=True,
        dropout_rate=rate,
        l2_reg=0.0,  # Zero L2 to isolate dropout effect
        optimizer='adam',
        lr=0.001
    )
    
    train_accs = []
    val_accs = []
    
    batch_size = 256
    for epoch in range(epochs):
        indices = np.random.permutation(X_tr.shape[0])
        X_tr_sh = X_tr[indices]
        y_tr_sh = y_tr[indices]
        
        for start in range(0, X_tr.shape[0], batch_size):
            end = start + batch_size
            xb = X_tr_sh[start:end]
            yb = y_tr_sh[start:end]
            
            y_pred = mlp.forward(xb, training=True)
            grads = mlp.backward(xb, yb)
            mlp.update_parameters(grads)
            
        # Evaluate on Train (without dropout mask)
        tr_acc, *_ = mlp.evaluate(X_tr, y_tr)
        # Evaluate on Val
        v_acc, *_ = mlp.evaluate(X_val, y_val)
        
        train_accs.append(tr_acc)
        val_accs.append(v_acc)
        
    return train_accs, val_accs


if __name__ == '__main__':
    print("=" * 60)
    print("LexiCoach — Dropout Study")
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
    
    rates = [0.0, 0.2, 0.3, 0.5]
    results = {}
    
    for r in rates:
        tr_accs, val_accs = train_with_dropout(r, X_train, y_train, X_val, y_val, epochs=25)
        results[r] = {
            'train': tr_accs,
            'val': val_accs
        }
        
    # Save the plot
    os.makedirs("results/plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    
    colors = {0.0: '#E74C3C', 0.2: '#F1C40F', 0.3: '#2ECC71', 0.5: '#3498DB'}
    for r in rates:
        # Plot train-test (val) gap
        gap = np.array(results[r]['train']) - np.array(results[r]['val'])
        plt.plot(gap, label=f'Dropout {r} (Train-Val Gap)', color=colors[r], linewidth=2)
        
    plt.title('Dropout Study: Train-Validation Accuracy Gap')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/10_dropout_study.png', dpi=150)
    plt.close()
    print("\nSaved comparison plot to -> results/plots/10_dropout_study.png")
