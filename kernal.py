import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# dataset A: linearly separable
X_linear, y_linear = make_classification(n_samples=400, n_features=2, n_informative=2, 
                                        n_redundant=0, n_clusters_per_class=1, 
                                        random_state=42, class_sep=2.0)

# dataset B: circles
X_circles, y_circles = make_circles(n_samples=400, factor=0.5, noise=0.1, random_state=42)

datasets = [
    ("Linear Data (Simple)", X_linear, y_linear),
    ("Non-Linear Data (Circles)", X_circles, y_circles),
]

models = {
    "Lasso (L1)": LogisticRegression(penalty='l1', solver='liblinear', C=1.0),
    "Linear SVM": SVC(kernel='linear', C=1.0),
    "RBF SVM (Kernel)": SVC(kernel='rbf', C=1.0, gamma='scale') 
}

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))

plt.subplots_adjust(wspace=0.2, hspace=0.3)

for row_idx, (ds_name, X, y) in enumerate(datasets):
    # normalize
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    h = .02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    for col_idx, (model_name, clf) in enumerate(models.items()):
        ax = axes[row_idx, col_idx]
        
        # train and evaluate
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        
        Z = Z.reshape(xx.shape)
        
        # plot
        ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=20)
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        
        ax.set_title(f"{model_name}\nAcc: {score:.2f}", color='black', fontweight='normal', fontsize=14)

        if col_idx == 0:
            ax.set_ylabel(ds_name, fontsize=14, fontweight='bold', labelpad=10)

plt.suptitle("Model Performance: Linear vs. Non-Linear Datasets", fontsize=16, y=0.95)
plt.savefig('exp2.png', dpi=300, bbox_inches='tight')
plt.show()