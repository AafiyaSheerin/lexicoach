import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("mlp", "04_mlp.py")
mlp_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp_mod)
MLP = mlp_mod.MLP

# ══════════════════════════════════════════════════════════════
# 05_backprop_manual.py
# LexiCoach — Backpropagation Verification (Manual vs Numerical)
# ══════════════════════════════════════════════════════════════

def compute_numerical_gradients(mlp, X, y, epsilon=1e-5):
    """
    Computes numerical gradients using two-sided finite differences.
    """
    num_grads = {
        'dW': [np.zeros_like(w) for w in mlp.W],
        'db': [np.zeros_like(b) for b in mlp.b],
        'dgamma': [np.zeros_like(g) if g is not None else None for g in mlp.gamma],
        'dbeta': [np.zeros_like(b) if b is not None else None for b in mlp.beta]
    }
    
    # 1. Weights W
    for i in range(len(mlp.W)):
        W_ref = mlp.W[i]
        for row in range(W_ref.shape[0]):
            for col in range(W_ref.shape[1]):
                old_val = W_ref[row, col]
                
                # J(theta + epsilon)
                W_ref[row, col] = old_val + epsilon
                y_plus = mlp.forward(X, training=True)
                loss_plus = mlp.compute_loss(y_plus, y)
                
                # J(theta - epsilon)
                W_ref[row, col] = old_val - epsilon
                y_minus = mlp.forward(X, training=True)
                loss_minus = mlp.compute_loss(y_minus, y)
                
                # Reset
                W_ref[row, col] = old_val
                
                # Finite difference formula
                num_grads['dW'][i][row, col] = (loss_plus - loss_minus) / (2.0 * epsilon)
                
    # 2. Biases b
    for i in range(len(mlp.b)):
        b_ref = mlp.b[i]
        for col in range(b_ref.shape[1]):
            old_val = b_ref[0, col]
            
            # J(theta + epsilon)
            b_ref[0, col] = old_val + epsilon
            y_plus = mlp.forward(X, training=True)
            loss_plus = mlp.compute_loss(y_plus, y)
            
            # J(theta - epsilon)
            b_ref[0, col] = old_val - epsilon
            y_minus = mlp.forward(X, training=True)
            loss_minus = mlp.compute_loss(y_minus, y)
            
            # Reset
            b_ref[0, col] = old_val
            
            num_grads['db'][i][0, col] = (loss_plus - loss_minus) / (2.0 * epsilon)
            
    # 3. BatchNorm parameters gamma & beta
    if mlp.use_batchnorm:
        for i in range(len(mlp.gamma)):
            g_ref = mlp.gamma[i]
            if g_ref is not None:
                for col in range(g_ref.shape[1]):
                    old_val = g_ref[0, col]
                    
                    g_ref[0, col] = old_val + epsilon
                    y_plus = mlp.forward(X, training=True)
                    loss_plus = mlp.compute_loss(y_plus, y)
                    
                    g_ref[0, col] = old_val - epsilon
                    y_minus = mlp.forward(X, training=True)
                    loss_minus = mlp.compute_loss(y_minus, y)
                    
                    g_ref[0, col] = old_val
                    num_grads['dgamma'][i][0, col] = (loss_plus - loss_minus) / (2.0 * epsilon)
                    
            bt_ref = mlp.beta[i]
            if bt_ref is not None:
                for col in range(bt_ref.shape[1]):
                    old_val = bt_ref[0, col]
                    
                    bt_ref[0, col] = old_val + epsilon
                    y_plus = mlp.forward(X, training=True)
                    loss_plus = mlp.compute_loss(y_plus, y)
                    
                    bt_ref[0, col] = old_val - epsilon
                    y_minus = mlp.forward(X, training=True)
                    loss_minus = mlp.compute_loss(y_minus, y)
                    
                    bt_ref[0, col] = old_val
                    num_grads['dbeta'][i][0, col] = (loss_plus - loss_minus) / (2.0 * epsilon)
                    
    return num_grads


def compare_gradients(num_g, ana_g, prefix=""):
    """
    Computes relative errors between numerical and analytical gradients.
    """
    print(f"\n{prefix} Gradient Verification:")
    print("-" * 60)
    
    # 1. Check Weights
    for i in range(len(num_g['dW'])):
        num = num_g['dW'][i]
        ana = ana_g['dW'][i]
        diff = np.linalg.norm(num - ana)
        denom = max(np.linalg.norm(num), np.linalg.norm(ana)) + 1e-15
        rel_error = diff / denom
        if np.linalg.norm(num) < 1e-7 and np.linalg.norm(ana) < 1e-7:
            rel_error = 0.0
        status = "PASS" if rel_error < 1e-6 else "FAIL"
        print(f"  Layer {i+1} W  | Rel Error: {rel_error:.2e} | Status: {status}")
        
    # 2. Check Biases
    for i in range(len(num_g['db'])):
        num = num_g['db'][i]
        ana = ana_g['db'][i]
        diff = np.linalg.norm(num - ana)
        denom = max(np.linalg.norm(num), np.linalg.norm(ana)) + 1e-15
        rel_error = diff / denom
        if np.linalg.norm(num) < 1e-7 and np.linalg.norm(ana) < 1e-7:
            rel_error = 0.0
        status = "PASS" if rel_error < 1e-6 else "FAIL"
        print(f"  Layer {i+1} b  | Rel Error: {rel_error:.2e} | Status: {status}")
        
    # 3. Check BatchNorm parameters
    if 'dgamma' in num_g:
        for i in range(len(num_g['dgamma'])):
            num = num_g['dgamma'][i]
            ana = ana_g['dgamma'][i]
            if num is not None:
                diff = np.linalg.norm(num - ana)
                denom = max(np.linalg.norm(num), np.linalg.norm(ana)) + 1e-15
                rel_error = diff / denom
                if np.linalg.norm(num) < 1e-7 and np.linalg.norm(ana) < 1e-7:
                    rel_error = 0.0
                status = "PASS" if rel_error < 1e-6 else "FAIL"
                print(f"  Layer {i+1} BN gamma | Rel Error: {rel_error:.2e} | Status: {status}")
                
            num_b = num_g['dbeta'][i]
            ana_b = ana_g['dbeta'][i]
            if num_b is not None:
                diff = np.linalg.norm(num_b - ana_b)
                denom = max(np.linalg.norm(num_b), np.linalg.norm(ana_b)) + 1e-15
                rel_error = diff / denom
                if np.linalg.norm(num_b) < 1e-7 and np.linalg.norm(ana_b) < 1e-7:
                    rel_error = 0.0
                status = "PASS" if rel_error < 1e-6 else "FAIL"
                print(f"  Layer {i+1} BN beta  | Rel Error: {rel_error:.2e} | Status: {status}")


if __name__ == '__main__':
    print("=" * 60)
    print("LexiCoach — Backpropagation Verification")
    print("=" * 60)
    
    # Generate mock inputs
    np.random.seed(42)
    X = np.random.randn(5, 10)  # 5 samples, 10 features
    y = np.random.randint(0, 2, size=(5, 1)).astype(float)
    
    # ── Test Case 1: Standard MLP (No BatchNorm, No Dropout) ────
    mlp_std = MLP(
        layer_sizes=[10, 8, 4, 1],
        activation='relu',
        use_batchnorm=False,
        dropout_rate=0.0,
        l2_reg=0.01
    )
    
    # Compute analytical gradients
    _ = mlp_std.forward(X, training=True)
    ana_std = mlp_std.backward(X, y)
    
    # Compute numerical gradients
    num_std = compute_numerical_gradients(mlp_std, X, y)
    
    # Compare
    compare_gradients(num_std, ana_std, "Standard MLP (No BatchNorm)")
    
    # ── Test Case 2: Batch Normalization MLP ────────────────────
    mlp_bn = MLP(
        layer_sizes=[10, 8, 4, 1],
        activation='relu',
        use_batchnorm=True,
        dropout_rate=0.0,
        l2_reg=0.01
    )
    
    # Compute analytical gradients
    _ = mlp_bn.forward(X, training=True)
    ana_bn = mlp_bn.backward(X, y)
    
    # Compute numerical gradients
    num_bn = compute_numerical_gradients(mlp_bn, X, y)
    
    # Compare
    compare_gradients(num_bn, ana_bn, "BatchNorm MLP")
    
    print("\n" + "=" * 60)
    print("BACKPROPAGATION VERIFICATION COMPLETE")
    print("=" * 60)
