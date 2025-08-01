# Online News Popularity — End-to-End Regression (Baseline LR → Tuned Random Forest)

Predicts article **shares** to inform content strategy using a reproducible ML pipeline. Trained on the **Online News Popularity** dataset (**~39k rows, 61 features**) with strong preprocessing, feature selection, and cross-validation.

## What’s inside
- **Data & Split:** ~39k articles, 61 predictors; **70/30 train–test** with **5-fold CV**.
- **Preprocessing:** missing-value checks, scaling (**StandardScaler**), **log1p target transform** to handle heavy right tail.
- **Feature Selection:** **SelectKBest(f_regression)** to keep top predictors.
- **Models:** **Linear Regression baseline** → **Random Forest** with grid search (**n_estimators=300, max_depth=20, max_features='sqrt'**).
- **Evaluation:** RMSE/MAE/**R²** on **log1p(shares)**; feature-importance insights for interpretability.

## Results (Test)
- **Random Forest (tuned):** **RMSE 0.857**, **MAE 0.637**, **R² 0.155** *(log-scale)*  
- Compared against **Linear Regression baseline** under identical CV protocol.
- Included feature importance to surface drivers of engagement.

**Code**
