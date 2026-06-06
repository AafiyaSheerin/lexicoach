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
# experiments/08_relu_vs_sigmoid.py
# LexiCoach — Vanishing Gradient Study (ReLU vs Sigmoid)
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("LexiCoach — Vanishing Gradient Study")
    print("=" * 60)
    
    # Load dataset
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        
    X_train = data['X_train']
    y_train = data['y_train']
    
    # We choose a batch of size 256
    np.random.seed(42)
    xb = X_train[:256]
    yb = y_train[:256]
    
    # Initialize ReLU Model (Standard initialization)
    mlp_relu = MLP(
        layer_sizes=[X_train.shape[1], 64, 32, 1],
        activation='relu',
        use_batchnorm=False,
        dropout_rate=0.0
    )
    
    # Initialize Sigmoid Model (Standard initialization)
    mlp_sig = MLP(
        layer_sizes=[X_train.shape[1], 64, 32, 1],
        activation='sigmoid',
        use_batchnorm=False,
        dropout_rate=0.0
    )
    
    relu_norms = []
    sig_norms = []
    
    print("Tracking gradient norms of the first hidden layer weights (dW1) over 30 updates...")
    
    for step in range(30):
        # 1. ReLU Forward + Backward
        y_pred_r = mlp_relu.forward(xb, training=True)
        grads_r = mlp_relu.backward(xb, yb)
        # Gradient norm of W1
        norm_r = np.linalg.norm(grads_r['dW'][0])
        relu_norms.append(norm_r)
        mlp_relu.update_parameters(grads_r)
        
        # 2. Sigmoid Forward + Backward
        y_pred_s = mlp_sig.forward(xb, training=True)
        grads_s = mlp_sig.backward(xb, yb)
        # Gradient norm of W1
        norm_s = np.linalg.norm(grads_s['dW'][0])
        sig_norms.append(norm_s)
        mlp_sig.update_parameters(grads_s)
        
        if step % 5 == 0 or step == 29:
            print(f"  Step {step:2d} | ReLU dW1 Norm: {norm_r:.6f} | Sigmoid dW1 Norm: {norm_s:.6f}")
            
    print("\n" + "=" * 60)
    print("SUMMARY OF STUDY")
    print("=" * 60)
    print(f"  Mean ReLU dW1 Norm     : {np.mean(relu_norms):.6f}")
    print(f"  Mean Sigmoid dW1 Norm  : {np.mean(sig_norms):.6f}")
    print(f"  Ratio (ReLU/Sigmoid)   : {np.mean(relu_norms)/np.mean(sig_norms):.1f}x larger gradients")
    print("=" * 60)
    
    # Save the plot
    os.makedirs("results/plots", exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(relu_norms, label='ReLU Gradient Norm', color='#E74C3C', linewidth=2.5)
    plt.plot(sig_norms, label='Sigmoid Gradient Norm', color='#3498DB', linewidth=2.5)
    plt.yscale('log')
    plt.title('Vanishing Gradient Proof: ReLU vs Sigmoid (dW1 Norm)')
    plt.xlabel('Optimization Step')
    plt.ylabel('L2 Gradient Norm (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/08_relu_vs_sigmoid.png', dpi=150)
    plt.close()
    print("Plot saved -> results/plots/08_relu_vs_sigmoid.png")
