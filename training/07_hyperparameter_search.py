import os
import sys
import pickle
import numpy as np
import importlib.util

# Add root directory to path to import 04_mlp.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
spec = importlib.util.spec_from_file_location("mlp", os.path.abspath(os.path.join(os.path.dirname(__file__), "../04_mlp.py")))
mlp_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp_mod)
MLP = mlp_mod.MLP

# ══════════════════════════════════════════════════════════════
# training/07_hyperparameter_search.py
# LexiCoach — Hyperparameter Grid Search
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("LexiCoach — Hyperparameter Grid Search")
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
    
    # Grid configuration
    lrs = [0.01, 0.001]
    hidden_configs = [
        [X_train.shape[1], 32, 16, 1],
        [X_train.shape[1], 64, 32, 1]
    ]
    dropouts = [0.0, 0.3]
    
    best_f1 = 0.0
    best_config = {}
    
    print(f"Starting grid search over {len(lrs) * len(hidden_configs) * len(dropouts)} combinations...")
    print(f"{'LR':<8} {'Hidden Sizes':<16} {'Dropout':<8} | {'Val Acc':<8} {'Val F1':<8}")
    print("-" * 60)
    
    for lr in lrs:
        for layers in hidden_configs:
            for dob in dropouts:
                # Train small number of epochs (15) to speed up grid search
                mlp = MLP(
                    layer_sizes=layers,
                    activation='relu',
                    use_batchnorm=True,
                    dropout_rate=dob,
                    l2_reg=0.001,
                    optimizer='adam',
                    lr=lr
                )
                
                # Simple training loop for 15 epochs
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
                val_acc, _, _, val_f1, *_ = mlp.evaluate(X_val, y_val, threshold=0.5)
                
                # Format hidden size representation
                layers_str = "-".join(map(str, layers[1:-1]))
                print(f"{lr:<8} {layers_str:<16} {dob:<8.1f} | {val_acc:<7.2f}% {val_f1:<7.2f}%")
                
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_config = {
                        'lr': lr,
                        'layers': layers,
                        'dropout': dob,
                        'accuracy': val_acc,
                        'f1': val_f1
                    }
                    
    print("=" * 60)
    print("BEST GRID SEARCH CONFIGURATION:")
    print("=" * 60)
    print(f"  Learning Rate : {best_config['lr']}")
    print(f"  Hidden Layers : {best_config['layers'][1:-1]}")
    print(f"  Dropout Rate  : {best_config['dropout']}")
    print(f"  Val Accuracy  : {best_config['accuracy']:.2f}%")
    print(f"  Val F1 Score  : {best_config['f1']:.2f}%")
    print("=" * 60)
