import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


n_samples = 150     
n_informative = 10  # effective features
dims = [100, 500, 1000, 1500, 2000, 3000]

results = {
    'svm_acc': [], 'mlp_acc': [], 'ridge_acc': [], 'lasso_acc': [],
    'svm_cplx': [], 'mlp_cplx': [], 'ridge_cplx': [], 'lasso_cplx': []
}

print(f"Running Experiment: 4 Models Comparison (N={n_samples})")
print("=" * 140)
print(f"{'Dim':<6} | {'SVM Acc':<8} | {'MLP Acc':<8} | {'Ridge Acc':<9} | {'Lasso Acc':<9} | {'SVM SVs':<8} | {'MLP Param':<10} | {'Ridge Param':<11} | {'Lasso Param':<11}")
print("-" * 140)

for d in dims:
    # generate data
    X, y = make_classification(n_samples=n_samples, n_features=d, 
                                n_informative=n_informative, n_redundant=0, 
                                n_classes=2, random_state=42, shuffle=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # --- Model 1: SVM ---
    clf_svm = SVC(kernel='linear', C=0.1)
    clf_svm.fit(X_train, y_train)
    results['svm_acc'].append(clf_svm.score(X_test, y_test))
    # complexity = number of support vectors
    results['svm_cplx'].append(np.sum(clf_svm.n_support_))
    
    # --- Model 2: MLP  ---
    clf_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    clf_mlp.fit(X_train, y_train)
    results['mlp_acc'].append(clf_mlp.score(X_test, y_test))
    # complexity = total number of parameters (weights + biases)
    n_params_mlp = sum(w.size for w in clf_mlp.coefs_) + sum(b.size for b in clf_mlp.intercepts_)
    results['mlp_cplx'].append(n_params_mlp)

    # --- Model 3: Ridge (L2 LogReg) ---
    clf_ridge = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=1000)
    clf_ridge.fit(X_train, y_train)
    results['ridge_acc'].append(clf_ridge.score(X_test, y_test))
    # complexity = total number of all feature weights
    results['ridge_cplx'].append(clf_ridge.coef_.size + clf_ridge.intercept_.size)
    
    # --- Model 4: Lasso (L1 LogReg) ---
    clf_lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000)
    clf_lasso.fit(X_train, y_train)
    results['lasso_acc'].append(clf_lasso.score(X_test, y_test))
    # complexity = number of non-zero weights
    n_nonzero = np.sum(np.abs(clf_lasso.coef_) > 1e-5) + clf_lasso.intercept_.size
    results['lasso_cplx'].append(n_nonzero)
    
    print(f"{d:<6} | {results['svm_acc'][-1]:.4f}   | {results['mlp_acc'][-1]:.4f}   | {results['ridge_acc'][-1]:.4f}    | {results['lasso_acc'][-1]:.4f}    | {results['svm_cplx'][-1]:<8} | {results['mlp_cplx'][-1]:<10} | {results['ridge_cplx'][-1]:<11} | {results['lasso_cplx'][-1]:<11}")

# plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- Plot 1: Accuracy ---
ax1.plot(dims, results['svm_acc'], 'o-', linewidth=3, label='SVM', color='#1f77b4') # Blue
ax1.plot(dims, results['lasso_acc'], '*-', linewidth=2.5, label='Lasso', color='#2ca02c') # Green
ax1.plot(dims, results['ridge_acc'], 'x--', linewidth=2, label='Ridge', color='gray') # Gray
ax1.plot(dims, results['mlp_acc'], 's:', linewidth=2, label='MLP', color='#d62728', alpha=0.6) # Red

ax1.set_title(f'Prediction Accuracy vs. Dimension', fontsize=14)
ax1.set_xlabel('Feature Dimension (D)', fontsize=12)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=11)

# --- Plot 2: Complexity ---
ax2.set_yscale('log')

ax2.plot(dims, results['svm_cplx'], 'o-', linewidth=3, label='SVM', color='#1f77b4')
ax2.plot(dims, results['lasso_cplx'], '*-', linewidth=2.5, label='Lasso', color='#2ca02c')
ax2.plot(dims, results['ridge_cplx'], 'x--', linewidth=2, label='Ridge', color='gray')
ax2.plot(dims, results['mlp_cplx'], 's:', linewidth=2, label='MLP', color='#d62728', alpha=0.6)

ax2.set_title(f'Effective Model Complexity (Log Scale)', fontsize=14)
ax2.set_xlabel('Feature Dimension (D)', fontsize=12)
ax2.set_ylabel('Number of Effective Parameters', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6, which="both")
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('exp1.png', dpi=300)
plt.show()