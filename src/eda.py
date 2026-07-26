"""
Step 5: Exploratory Analysis & Modeling Patient Health Behavior Patterns
==========================================================================
Produces:
  - A correlation heatmap of raw (pre-scaling) features.
  - Class-conditional distribution plots for the top predictive features
    (diseased vs. non-diseased), which is the "behavior pattern" view --
    e.g. does glucose separate diabetics from non-diabetics, does exercise-
    induced angina separate heart-disease patients from healthy ones.

Figures are saved to outputs/ as PNGs.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_correlation_heatmap(df: pd.DataFrame, target_col: str, title: str, filename: str):
    plt.figure(figsize=(9, 7))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=130)
    plt.close()
    return path


def plot_class_conditional_distributions(df: pd.DataFrame, target_col: str,
                                          top_features: list, title: str, filename: str):
    n = len(top_features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()
    for i, feat in enumerate(top_features):
        sns.kdeplot(data=df, x=feat, hue=target_col, common_norm=False,
                    fill=True, alpha=0.4, ax=axes[i])
        axes[i].set_title(feat)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path


if __name__ == "__main__":
    from data_loader import load_diabetes_data, load_heart_data
    from feature_selection import rank_features_anova
    from preprocessing import preprocess_diabetes, preprocess_heart

    diabetes_raw = load_diabetes_data(verbose=False)
    p1 = plot_correlation_heatmap(diabetes_raw, "Outcome",
                                   "Diabetes: Feature Correlation Heatmap",
                                   "diabetes_correlation_heatmap.png")

    Xd, yd, _, _ = preprocess_diabetes(diabetes_raw)
    top_d = rank_features_anova(Xd, yd).head(6).index.tolist()
    diabetes_raw_labeled = diabetes_raw.copy()
    diabetes_raw_labeled["Outcome"] = diabetes_raw_labeled["Outcome"].map({0: "No Diabetes", 1: "Diabetes"})
    p2 = plot_class_conditional_distributions(diabetes_raw_labeled, "Outcome", top_d,
                                               "Diabetes: Behavior Pattern by Outcome",
                                               "diabetes_behavior_patterns.png")

    heart_raw = load_heart_data(verbose=False)
    p3 = plot_correlation_heatmap(heart_raw, "target",
                                   "Heart Disease: Feature Correlation Heatmap",
                                   "heart_correlation_heatmap.png")

    Xh, yh, _, _ = preprocess_heart(heart_raw)
    top_h = rank_features_anova(Xh, yh).head(6).index.tolist()
    heart_raw_labeled = heart_raw.copy()
    heart_raw_labeled["target"] = heart_raw_labeled["target"].map({0: "No Disease", 1: "Disease"})
    p4 = plot_class_conditional_distributions(heart_raw_labeled, "target", top_h,
                                               "Heart Disease: Behavior Pattern by Target",
                                               "heart_behavior_patterns.png")

    print("Saved:", p1, p2, p3, p4, sep="\n  ")
