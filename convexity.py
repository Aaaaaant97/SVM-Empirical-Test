import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=200, noise=0.3, random_state=42)

# normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_runs = 10
h = 0.02
x_min, x_max = -6, 6
y_min, y_max = -6, 6

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# MLP
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=30, alpha=0.6)
axes[0].set_title("MLP (Neural Network)", fontsize=12)
axes[0].set_xlim(x_min, x_max)
axes[0].set_ylim(y_min, y_max)
print("Training MLPs...")
for i in range(n_runs):
    # different random init
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=i)
    mlp.fit(X_scaled, y)
    
    if hasattr(mlp, "decision_function"):
        Z = mlp.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = mlp.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
    Z = Z.reshape(xx.shape)
    axes[0].contour(xx, yy, Z, levels=[0.5], colors='red', alpha=0.3, linewidths=1.5)

# SVM
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=30, alpha=0.6)
axes[1].set_title("SVM", fontsize=12)
axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)
print("Training SVMs...")
for i in range(n_runs):
    # different random init
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=i) 
    svm.fit(X_scaled, y)
    
    if hasattr(svm, "decision_function"):
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = svm.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        
    Z = Z.reshape(xx.shape)
    axes[1].contour(xx, yy, Z, levels=[0], colors='blue', alpha=0.3, linewidths=1.5)

plt.tight_layout()
plt.savefig('exp3_convex.png', dpi=300)
plt.show()