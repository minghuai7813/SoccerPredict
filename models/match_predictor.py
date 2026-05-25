"""
World Cup match outcome predictor with market alpha analysis.
世界杯比赛结果预测器，含市场 alpha 分析。

Model design / 模型设计:
  - Task 1: 3-class classification (home win / draw / away win).
    任务 1：三分类（主队赢 / 平局 / 客队赢）。
  - Task 2: Regression (predict goal difference).
    任务 2：回归（预测净胜球）。
  - Task 3: Alpha analysis — compare model vs Elo-based "market" prediction.
    任务 3：Alpha 分析——对比模型与 Elo 基准（市场共识）的分歧。
    If we predict the same as Elo (market), we have no edge.
    如果我们和 Elo（市场）预测一致，说明没有优势。
    Value exists only where we DISAGREE with the market AND are correct.
    价值只存在于我们与市场意见不同、且我们是对的地方。

  - Algorithm: Random Forest (baseline).
  - Evaluation: Leave-One-Out cross-validation.

Usage / 用法:
    python -m models.match_predictor
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def train_and_evaluate() -> dict:
    """
    End-to-end: build dataset → train model → evaluate via LOO-CV.
    端到端：构建数据集 → 训练模型 → 通过 LOO-CV 评估。

    Returns
    -------
    dict
        Evaluation metrics and the trained model artifacts.
    """
    from features.match_dataset import build_match_dataset

    print("=" * 60)
    print("  PROJECT ORACLE — Match Predictor v0.1 (Baseline)")
    print("=" * 60)

    X, y, meta = build_match_dataset()

    if X.empty:
        print("[Model] No data to train on. Aborting.")
        return {}

    # Replace any remaining NaN/inf with 0 (defensive).
    # 防御性处理：把残余的 NaN/inf 替换为 0。
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    feature_names = X.columns.tolist()

    # Scale features — important for interpretation and some algorithms.
    # 标准化特征——对解释性和某些算法很重要。
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_names,
        index=X.index,
    )

    # ----------------------------------------------------------------
    # Task 1: Classification (home win / draw / away win)
    # 任务 1：三分类
    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Task 1] 3-Class Classification (Home Win / Draw / Away Win)")
    print("-" * 60)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
    )

    loo = LeaveOneOut()
    y_pred_clf = cross_val_predict(clf, X_scaled, y, cv=loo)

    acc = accuracy_score(y, y_pred_clf)
    print(f"\n  LOO-CV Accuracy: {acc:.1%}")
    print(f"  (Random baseline for 3-class: ~33.3%)")
    print(f"\n  Classification Report:")
    label_map = {1: "Home Win", 0: "Draw", -1: "Away Win"}
    print(classification_report(
        y, y_pred_clf,
        target_names=[label_map[l] for l in sorted(y.unique())],
        zero_division=0,
    ))

    # Show some example predictions.
    # 展示一些预测示例。
    meta_display = meta[["home_team", "away_team", "home_score", "away_score"]].copy()
    meta_display["actual"] = y.map(label_map)
    meta_display["predicted"] = pd.Series(y_pred_clf).map(label_map)
    meta_display["correct"] = (y.values == y_pred_clf)

    print("  Sample Predictions (first 15 matches):")
    print(meta_display.head(15).to_string(index=False))

    wrong = meta_display[~meta_display["correct"]]
    print(f"\n  Incorrect predictions: {len(wrong)}/{len(meta_display)}")

    # ----------------------------------------------------------------
    # Task 2: Regression (predict goal difference)
    # 任务 2：回归（预测净胜球）
    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Task 2] Regression (Predict Goal Difference)")
    print("-" * 60)

    y_reg = meta["goal_diff"]
    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42,
    )

    y_pred_reg = cross_val_predict(reg, X_scaled, y_reg, cv=loo)
    mae = mean_absolute_error(y_reg, y_pred_reg)
    print(f"\n  LOO-CV MAE (Mean Absolute Error): {mae:.2f} goals")

    # Derive classification from predicted goal diff and measure accuracy.
    # 从预测净胜球反推胜平负分类，测量准确率。
    y_pred_from_reg = pd.Series(y_pred_reg).apply(
        lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0)
    )
    acc_from_reg = accuracy_score(y, y_pred_from_reg)
    print(f"  Derived classification accuracy: {acc_from_reg:.1%}")

    # ----------------------------------------------------------------
    # Feature importance (from a final full-data fit)
    # 特征重要性（从全量数据训练中提取）
    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Feature Importance] Top 15 (from full-data Random Forest)")
    print("-" * 60)

    clf.fit(X_scaled, y)
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(15)
    for feat, imp in top_features.items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<40s} {imp:.4f}  {bar}")

    # ----------------------------------------------------------------
    # Task 3: Alpha analysis — model vs Elo "market" baseline
    # 任务 3：Alpha 分析——模型 vs Elo "市场"基准
    # ----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Task 3] Alpha Analysis: Our Model vs Market (Elo Baseline)")
    print("-" * 60)

    from extractors.elo_scraper import get_pre_wc_elo
    elo_data = get_pre_wc_elo()

    # Elo win-probability formula: E(A) = 1 / (1 + 10^((Rb-Ra)/400))
    # Elo 胜率公式：基于两队 Elo 差计算期望胜率。
    # We split into 3 classes using thresholds derived from WC draw rate (~23%).
    # 根据世界杯平局率（约 23%）设定阈值拆分为三类。
    def elo_predict(home_team: str, away_team: str) -> int:
        ra = elo_data.get(home_team, {}).get("elo", 1500)
        rb = elo_data.get(away_team, {}).get("elo", 1500)
        home_win_prob = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        # ~60 Elo points of home advantage in neutral-venue WC.
        # 世界杯中立场地约 60 Elo 点的主场优势。
        home_win_prob_adj = 1.0 / (1.0 + 10 ** ((rb - ra - 60) / 400.0))
        if home_win_prob_adj > 0.45:
            return 1
        elif home_win_prob_adj < 0.35:
            return -1
        else:
            return 0

    def elo_win_prob(home_team: str, away_team: str) -> float:
        ra = elo_data.get(home_team, {}).get("elo", 1500)
        rb = elo_data.get(away_team, {}).get("elo", 1500)
        return 1.0 / (1.0 + 10 ** ((rb - ra - 60) / 400.0))

    elo_preds = []
    elo_probs = []
    for _, m in meta.iterrows():
        elo_preds.append(elo_predict(m["home_team"], m["away_team"]))
        elo_probs.append(elo_win_prob(m["home_team"], m["away_team"]))

    elo_preds = np.array(elo_preds)
    elo_acc = accuracy_score(y, elo_preds)

    # Get our model's LOO predicted probabilities.
    # 获取我们模型的 LOO 预测概率。
    y_pred_proba = cross_val_predict(clf, X_scaled, y, cv=loo, method="predict_proba")
    model_classes = sorted(y.unique())
    home_win_idx = model_classes.index(1) if 1 in model_classes else -1

    print(f"\n  Elo-only accuracy:    {elo_acc:.1%}  (= market consensus baseline)")
    print(f"  Our model accuracy:   {acc:.1%}")
    edge = acc - elo_acc
    if edge > 0:
        print(f"  Model edge:           +{edge:.1%}  (we beat the market)")
    elif edge < 0:
        print(f"  Model edge:           {edge:.1%}  (market is better)")
    else:
        print(f"  Model edge:           0%  (no difference)")

    # Build comparison table.
    # 构建对比表。
    alpha_df = meta[["home_team", "away_team", "home_score", "away_score"]].copy()
    alpha_df["actual"] = y.map(label_map).values
    alpha_df["elo_pred"] = pd.Series(elo_preds).map(label_map).values
    alpha_df["model_pred"] = pd.Series(y_pred_clf).map(label_map).values
    alpha_df["elo_home_prob"] = [f"{p:.0%}" for p in elo_probs]

    if home_win_idx >= 0:
        alpha_df["model_home_prob"] = [f"{p[home_win_idx]:.0%}" for p in y_pred_proba]

    alpha_df["elo_correct"] = (y.values == elo_preds)
    alpha_df["model_correct"] = (y.values == y_pred_clf)
    alpha_df["agree"] = (elo_preds == y_pred_clf)

    # Disagreement analysis — where the model differs from market.
    # 分歧分析——模型和市场意见不同的比赛。
    disagree = alpha_df[~alpha_df["agree"]]
    agree = alpha_df[alpha_df["agree"]]

    print(f"\n  Agreement with market: {len(agree)}/64 matches ({len(agree)/64:.0%})")
    print(f"  Disagreements:         {len(disagree)}/64 matches")

    if len(disagree) > 0:
        model_right_on_disagree = disagree["model_correct"].sum()
        elo_right_on_disagree = disagree["elo_correct"].sum()
        neither = len(disagree) - model_right_on_disagree - elo_right_on_disagree
        both_wrong_count = len(disagree) - model_right_on_disagree - elo_right_on_disagree

        print(f"\n  On disagreements ({len(disagree)} matches):")
        print(f"    Model correct, market wrong:  {model_right_on_disagree}")
        print(f"    Market correct, model wrong:  {elo_right_on_disagree}")
        print(f"    Both wrong:                   {both_wrong_count}")

        if model_right_on_disagree > elo_right_on_disagree:
            print(f"    >>> Our model has ALPHA on disagreements!")
        elif model_right_on_disagree < elo_right_on_disagree:
            print(f"    >>> Market wins on disagreements (no alpha yet)")
        else:
            print(f"    >>> Tied on disagreements")

        # Show the disagreement matches.
        # 展示分歧比赛。
        print(f"\n  Disagreement matches (model vs market):")
        cols = ["home_team", "away_team", "actual", "elo_pred",
                "model_pred", "elo_home_prob"]
        if "model_home_prob" in disagree.columns:
            cols.append("model_home_prob")
        cols += ["model_correct"]
        print(disagree[cols].to_string(index=False))

    # Upset detection — matches where the favorite lost.
    # 冷门检测——热门输了的比赛。
    print(f"\n  --- Upset Detection (Dark Horses) ---")
    upsets = alpha_df[~alpha_df["elo_correct"]].copy()
    print(f"  Elo got {len(upsets)} matches wrong (potential upsets).")
    model_caught = upsets[upsets["model_correct"]]
    print(f"  Our model correctly predicted {len(model_caught)}/{len(upsets)} of these upsets:")
    if len(model_caught) > 0:
        cols = ["home_team", "away_team", "actual", "elo_pred", "model_pred"]
        print(model_caught[cols].to_string(index=False))

    print("\n" + "=" * 60)
    print("  Analysis complete.")
    print("=" * 60)

    return {
        "accuracy_clf": acc,
        "accuracy_elo": elo_acc,
        "mae_reg": mae,
        "accuracy_from_reg": acc_from_reg,
        "top_features": top_features,
        "classifier": clf,
        "regressor": reg,
        "scaler": scaler,
        "alpha_df": alpha_df,
    }


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()

    train_and_evaluate()
