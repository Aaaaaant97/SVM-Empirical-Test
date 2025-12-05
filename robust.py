import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# primary clean data
X, y_clean = make_moons(n_samples=200, noise=0.3, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# make noisy labels by flipping 15% of the labels
y_noisy = y_clean.copy()
np.random.seed(42)
n_noise = int(0.15 * len(y_clean))
noise_indices = np.random.choice(len(y_clean), n_noise, replace=False)
y_noisy[noise_indices] = 1 - y_noisy[noise_indices]


x_min, x_max = -4, 4
y_min, y_max = -4, 4
h = 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

fig, axes = plt.subplots(2, 2, figsize=(14, 11)) 

def train_and_plot(ax, model_class, model_params, X_train, y_train, y_true, title_prefix, contour_color, noise_idx=None):
    # train the model
    print(f"Training {title_prefix}...")
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    
    # calculate accuracies
    train_acc = model.score(X_train, y_train)
    true_acc = model.score(X_train, y_true)
    
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=30, alpha=0.6)
    
    # denote mislabeled points if any
    if noise_idx is not None:
        ax.scatter(X_train[noise_idx, 0], X_train[noise_idx, 1], 
                   facecolors='none', edgecolors='yellow', s=100, linewidth=2, label="Mislabeled Points")
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)

    # plot decision boundary
    if hasattr(model, "decision_function"):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        levels = [0]
    else:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        levels = [0.5]
    
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=levels, colors=contour_color, linewidths=2.5)

    full_title = (f"{title_prefix}\n"
                  f"Train Acc: {train_acc:.1%}\n"
                  f"Test Acc:  {true_acc:.1%}")
    
    ax.set_title(full_title, fontsize=11, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    

# parameters for models
mlp_params = {
    'hidden_layer_sizes': (400, 200),
    'activation': 'relu',
    'solver': 'lbfgs',
    'alpha': 0,
    'max_iter': 5000,
    'random_state': 42
}

svm_params = {
    'kernel': 'rbf',
    'C': 1.0,
    'probability': True,
    'random_state': 42
}

# clean data
train_and_plot(axes[0, 0], MLPClassifier, mlp_params, X_scaled, y_clean, y_clean,
               "MLP (Clean Data)", 'red')

train_and_plot(axes[0, 1], SVC, svm_params, X_scaled, y_clean, y_clean,
               "SVM (Clean Data)", 'blue')

# flipped noisy data
train_and_plot(axes[1, 0], MLPClassifier, mlp_params, X_scaled, y_noisy, y_clean,
               "MLP (Noisy Data) - Overfitting", 'red', noise_indices)

train_and_plot(axes[1, 1], SVC, svm_params, X_scaled, y_noisy, y_clean,
               "SVM (Noisy Data) - Robust", 'blue', noise_indices)

plt.tight_layout()
plt.savefig('exp4_robustness.png', dpi=300)
plt.show()