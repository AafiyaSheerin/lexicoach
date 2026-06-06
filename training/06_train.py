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
# training/06_train.py
# LexiCoach — SGD vs Adam Optimizer Study
# ══════════════════════════════════════════════════════════════

def train_model(optimizer_name, lr, X_train, y_train, X_val, y_val, epochs=50, batch_size=256):
    print(f"\nTraining MLP with {optimizer_name.upper()} (lr={lr})...")
    mlp = MLP(
        layer_sizes=[X_train.shape[1], 64, 32, 1],
        activation='relu',
        use_batchnorm=True,
        dropout_rate=0.2,
        l2_reg=0.001,
        optimizer=optimizer_name,
        lr=lr
    )
    
    losses = []
    val_f1s = []
    
    for epoch in range(epochs):
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
            
        losses.append(np.mean(epoch_losses))
        
        # Validation F1
        _, _, _, val_f1, *_ = mlp.evaluate(X_val, y_val, threshold=0.5)
        val_f1s.append(val_f1)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {losses[-1]:.4f} | Val F1: {val_f1s[-1]:.2f}%")
            
    return mlp, losses, val_f1s


if __name__ == '__main__':
    print("=" * 60)
    print("LexiCoach — SGD vs Adam Optimizer Study")
    print("=" * 60)
    
    # Load dataset
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        
    X_train_full = data['X_train']
    y_train_full = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    val_size = int(0.15 * len(X_train_full))
    X_val = X_train_full[:val_size]
    y_val = y_train_full[:val_size]
    X_train = X_train_full[val_size:]
    y_train = y_train_full[val_size:]
    
    epochs = 40
    batch_size = 256
    
    # Train SGD Model
    sgd_model, sgd_losses, sgd_f1s = train_model('sgd', 0.01, X_train, y_train, X_val, y_val, epochs, batch_size)
    
    # Train Adam Model
    adam_model, adam_losses, adam_f1s = train_model('adam', 0.001, X_train, y_train, X_val, y_val, epochs, batch_size)
    
    # Evaluate on Test Set
    print("\n" + "-" * 50)
    print("Test Set Comparison:")
    print("-" * 50)
    
    s_acc, s_prec, s_rec, s_f1, *_ = sgd_model.evaluate(X_test, y_test, threshold=0.5)
    print(f"  SGD  (lr=0.01)  | Acc: {s_acc:.2f}% | Prec: {s_prec:.2f}% | Rec: {s_rec:.2f}% | F1: {s_f1:.2f}%")
    
    a_acc, a_prec, a_rec, a_f1, *_ = adam_model.evaluate(X_test, y_test, threshold=0.5)
    print(f"  Adam (lr=0.001) | Acc: {a_acc:.2f}% | Prec: {a_prec:.2f}% | Rec: {a_rec:.2f}% | F1: {a_f1:.2f}%")
    print("-" * 50)
    
    # Plotting comparison curves
    os.makedirs("results/plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Optimizer Comparison: SGD vs Adam", fontsize=14)
    
    # Loss curves
    axes[0].plot(sgd_losses, label='SGD Loss', color='#F4A261', linewidth=2)
    axes[0].plot(adam_losses, label='Adam Loss', color='#2A9D8F', linewidth=2)
    axes[0].set_title('Training Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Validation F1 curves
    axes[1].plot(sgd_f1s, label='SGD Val F1', color='#F4A261', linewidth=2)
    axes[1].plot(adam_f1s, label='Adam Val F1', color='#2A9D8F', linewidth=2)
    axes[1].set_title('Validation F1 Score (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/06_optimizer_comparison.png', dpi=150)
    plt.close()
    print("\nSaved comparative plot to -> results/plots/06_optimizer_comparison.png")
