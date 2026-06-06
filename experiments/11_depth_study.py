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
# experiments/11_depth_study.py
# LexiCoach — Network Depth Study
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("LexiCoach — Network Depth Study")
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
    
    # Define hidden layer configs (depth 1 to 5)
    in_dim = X_train.shape[1]
    configs = {
        1: [in_dim, 64, 1],
        2: [in_dim, 64, 32, 1],
        3: [in_dim, 64, 32, 16, 1],
        4: [in_dim, 64, 32, 16, 8, 1],
        5: [in_dim, 64, 32, 16, 8, 4, 1]
    }
    
    depths = list(configs.keys())
    accuracies = []
    f1s = []
    
    print("Evaluating models with different layer depths...")
    for d in depths:
        layers = configs[d]
        print(f"  Training depth {d}hidden layers: {layers[1:-1]}...")
        
        mlp = MLP(
            layer_sizes=layers,
            activation='relu',
            use_batchnorm=True,
            dropout_rate=0.2,
            l2_reg=0.001,
            optimizer='adam',
            lr=0.001
        )
        
        # Train for 15 epochs
        batch_size = 256
        for epoch in range(15):
            indices = np.random.permutation(X_train.shape[0])
            X_tr_sh = X_train[indices]
            y_tr_sh = y_train[indices]
            
            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                xb = X_tr_sh[start:end]
                yb = y_tr_sh[start:end]
                
                y_pred = mlp.forward(xb, training=True)
                grads = mlp.backward(xb, yb)
                mlp.update_parameters(grads)
                
        # Evaluate on Val
        v_acc, _, _, v_f1, *_ = mlp.evaluate(X_val, y_val, threshold=0.5)
        accuracies.append(v_acc)
        f1s.append(v_f1)
        print(f"    Depth {d} | Val Accuracy: {v_acc:.2f}% | Val F1: {v_f1:.2f}%")
        
    # Save the plot
    os.makedirs("results/plots", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    
    color = '#8E44AD'
    ax1.set_xlabel('Network Depth (Number of Hidden Layers)')
    ax1.set_ylabel('Validation F1 Score (%)', color=color)
    line1 = ax1.plot(depths, f1s, marker='o', color=color, linewidth=2.5, label='Validation F1')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = '#2980B9'
    ax2.set_ylabel('Validation Accuracy (%)', color=color)
    line2 = ax2.plot(depths, accuracies, marker='s', color=color, linewidth=2.5, linestyle='--', label='Validation Acc')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Network Depth Study: F1 Score and Accuracy')
    plt.tight_layout()
    plt.savefig('results/plots/11_depth_study.png', dpi=150)
    plt.close()
    print("\nSaved comparative plot to -> results/plots/11_depth_study.png")
