"""
Iris Flower Classification
==========================
Classifies iris flowers into Setosa, Versicolor, or Virginica
using multiple ML models with full evaluation and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ──────────────────────────────────────────────

def load_and_explore():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("=" * 60)
    print("           IRIS FLOWER CLASSIFICATION")
    print("=" * 60)
    print(f"\nDataset shape : {df.shape}")
    print(f"Classes       : {list(iris.target_names)}")
    print(f"Samples/class : {dict(df['species'].value_counts())}")

    print("\n── First 5 rows ──")
    print(df.head().to_string(index=False))

    print("\n── Statistical summary ──")
    print(df.describe().round(2).to_string())

    print("\n── Missing values ──")
    print(df.isnull().sum().to_string())

    return iris, df


# ──────────────────────────────────────────────
# 2. VISUALIZE DATA
# ──────────────────────────────────────────────

def visualize_data(df):
    species_colors = {"setosa": "#3266ad", "versicolor": "#2a8a5e", "virginica": "#c1440e"}

    # --- Pair plot ---
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    fig.suptitle("Iris Dataset — Pairwise Feature Plot", fontsize=15, fontweight="bold", y=1.01)
    features = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    short = ["Sepal L", "Sepal W", "Petal L", "Petal W"]

    for i, fi in enumerate(features):
        for j, fj in enumerate(features):
            ax = axes[i][j]
            if i == j:
                for sp, grp in df.groupby("species"):
                    ax.hist(grp[fi], bins=15, alpha=0.6, color=species_colors[sp], label=sp)
            else:
                for sp, grp in df.groupby("species"):
                    ax.scatter(grp[fj], grp[fi], alpha=0.6, s=18, color=species_colors[sp])
            if j == 0:
                ax.set_ylabel(short[i], fontsize=8)
            if i == 3:
                ax.set_xlabel(short[j], fontsize=8)
            ax.tick_params(labelsize=6)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markersize=8, label=s)
               for s, c in species_colors.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("plot_pairwise.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("\n[Saved] plot_pairwise.png")

    # --- Box plots ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Feature Distribution by Species", fontsize=13, fontweight="bold")
    palette = list(species_colors.values())
    for ax, feat, lbl in zip(axes, features, short):
        data = [df[df["species"] == sp][feat].values for sp in ["setosa", "versicolor", "virginica"]]
        bp = ax.boxplot(data, patch_artist=True, labels=["Setosa", "Versic.", "Virgin."])
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(lbl, fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    plt.savefig("plot_boxplots.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[Saved] plot_boxplots.png")

    # --- Correlation heatmap ---
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot_correlation.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[Saved] plot_correlation.png")


# ──────────────────────────────────────────────
# 3. TRAIN MULTIPLE MODELS
# ──────────────────────────────────────────────

def build_models():
    return {
        "Decision Tree":        DecisionTreeClassifier(random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)":            SVC(probability=True, random_state=42),
        "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression":  LogisticRegression(max_iter=200, random_state=42),
    }


def train_and_evaluate(iris):
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    models = build_models()
    results = {}

    print("\n" + "=" * 60)
    print("           MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, clf in models.items():
        # Wrap in pipeline with scaling
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")

        results[name] = {
            "pipeline":  pipe,
            "accuracy":  acc,
            "cv_mean":   cv_scores.mean(),
            "cv_std":    cv_scores.std(),
            "y_pred":    y_pred,
        }
        print(f"\n{name}")
        print(f"  Test accuracy : {acc*100:.1f}%")
        print(f"  CV 5-fold     : {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    return results, X_train, X_test, y_train, y_test, X, y


# ──────────────────────────────────────────────
# 4. BEST MODEL — DETAILED REPORT
# ──────────────────────────────────────────────

def detailed_report(results, X_test, y_test, iris):
    best_name = max(results, key=lambda n: results[n]["cv_mean"])
    best = results[best_name]

    print("\n" + "=" * 60)
    print(f"  BEST MODEL: {best_name}  ({best['cv_mean']*100:.1f}% CV accuracy)")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(
        y_test, best["y_pred"],
        target_names=iris.target_names
    ))
    return best_name, best


# ──────────────────────────────────────────────
# 5. VISUALIZE RESULTS
# ──────────────────────────────────────────────

def visualize_results(results, X_test, y_test, iris, best_name, best):

    # --- Model comparison bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    accs  = [results[n]["cv_mean"] * 100 for n in names]
    stds  = [results[n]["cv_std"]  * 100 for n in names]
    colors = ["#3266ad" if n == best_name else "#b0bec5" for n in names]
    bars = ax.bar(names, accs, yerr=stds, capsize=5, color=colors,
                  edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.set_ylim(88, 102)
    ax.set_ylabel("CV Accuracy (%)", fontsize=11)
    ax.set_title("Model Comparison — 5-Fold Cross Validation", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="500")
    plt.tight_layout()
    plt.savefig("plot_model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[Saved] plot_model_comparison.png")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, best["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[Saved] plot_confusion_matrix.png")

    # --- Feature importance (tree-based) ---
    tree_models = ["Random Forest", "Gradient Boosting", "Decision Tree"]
    chosen = next((n for n in [best_name] + tree_models if n in results), None)
    if chosen:
        clf = results[chosen]["pipeline"].named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            fi = clf.feature_importances_
            feats = ["Sepal L", "Sepal W", "Petal L", "Petal W"]
            order = np.argsort(fi)
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh([feats[i] for i in order], fi[order],
                           color="#3266ad", alpha=0.85, edgecolor="white")
            ax.set_xlabel("Importance (Gini reduction)", fontsize=11)
            ax.set_title(f"Feature Importance — {chosen}", fontsize=13, fontweight="bold")
            for bar, val in zip(bars, fi[order]):
                ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=10)
            plt.tight_layout()
            plt.savefig("plot_feature_importance.png", dpi=120, bbox_inches="tight")
            plt.close()
            print("[Saved] plot_feature_importance.png")

    # --- Decision boundary (petal features) ---
    pipe = results[best_name]["pipeline"]
    X_2d = X_test[:, 2:4]   # petal length & width only
    scaler2 = StandardScaler().fit(X_2d)
    X_2d_s = scaler2.transform(X_2d)

    x_min, x_max = X_2d_s[:, 0].min() - 0.5, X_2d_s[:, 0].max() + 0.5
    y_min, y_max = X_2d_s[:, 1].min() - 0.5, X_2d_s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    clf2 = Pipeline([("scaler", StandardScaler()), ("clf",
           results[best_name]["pipeline"].named_steps["clf"].__class__(
               **{k: v for k, v in
                  results[best_name]["pipeline"].named_steps["clf"].get_params().items()
                  if k != "base_estimator"}
           ))])
    clf2.fit(X_test[:, 2:4], y_test)
    Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    bg_colors = ["#d0e4f7", "#c8e6d8", "#fad5c8"]
    pt_colors = ["#3266ad", "#2a8a5e", "#c1440e"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.get_cmap("Set1", 3))
    for i, sp in enumerate(iris.target_names):
        idx = y_test == i
        ax.scatter(X_2d_s[idx, 0], X_2d_s[idx, 1],
                   c=pt_colors[i], label=sp, edgecolors="white",
                   s=60, linewidths=0.5, zorder=3)
    ax.set_xlabel("Petal Length (scaled)", fontsize=11)
    ax.set_ylabel("Petal Width (scaled)", fontsize=11)
    ax.set_title(f"Decision Boundary — {best_name} (petal features)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("plot_decision_boundary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[Saved] plot_decision_boundary.png")


# ──────────────────────────────────────────────
# 6. HYPERPARAMETER TUNING
# ──────────────────────────────────────────────

def tune_random_forest(X, y):
    print("\n" + "=" * 60)
    print("     HYPERPARAMETER TUNING — Random Forest")
    print("=" * 60)
    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth":    [None, 3, 5],
        "clf__min_samples_split": [2, 5],
    }
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", RandomForestClassifier(random_state=42))])
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(X, y)
    print(f"Best params   : {gs.best_params_}")
    print(f"Best CV score : {gs.best_score_*100:.2f}%")
    return gs.best_estimator_


# ──────────────────────────────────────────────
# 7. PREDICT NEW SAMPLES
# ──────────────────────────────────────────────

def predict_new_samples(best_pipeline, iris):
    print("\n" + "=" * 60)
    print("         PREDICTING NEW FLOWER SAMPLES")
    print("=" * 60)
    samples = np.array([
        [5.1, 3.5, 1.4, 0.2],   # likely Setosa
        [6.3, 3.3, 4.7, 1.6],   # likely Versicolor
        [6.7, 3.0, 5.2, 2.3],   # likely Virginica
        [5.8, 2.7, 4.1, 1.0],   # borderline
    ])
    labels = ["Likely Setosa", "Likely Versicolor", "Likely Virginica", "Borderline"]
    preds  = best_pipeline.predict(samples)
    probas = best_pipeline.predict_proba(samples)
    print(f"\n{'Sample':<18} {'Predicted':<14} {'Setosa':>8} {'Versicolor':>12} {'Virginica':>10}")
    print("-" * 64)
    for lbl, pred, proba in zip(labels, preds, probas):
        sp = iris.target_names[pred]
        print(f"{lbl:<18} {sp:<14} {proba[0]*100:>7.1f}% {proba[1]*100:>11.1f}% {proba[2]*100:>9.1f}%")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    iris, df = load_and_explore()
    visualize_data(df)
    results, X_train, X_test, y_train, y_test, X, y = train_and_evaluate(iris)
    best_name, best = detailed_report(results, X_test, y_test, iris)
    visualize_results(results, X_test, y_test, iris, best_name, best)
    best_tuned = tune_random_forest(X, y)
    predict_new_samples(best["pipeline"], iris)
    print("\n✓ All done! Check the generated PNG plots in this directory.\n")