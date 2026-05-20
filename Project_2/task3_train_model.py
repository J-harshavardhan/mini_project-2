import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)

# ── Load cleaned data ────────────────────────────────────────────────────────
df = pd.read_csv(r"..\DataSet\cleaned_data.csv")
print(f"✔ Loaded cleaned data — {df.shape[0]} rows, {df.shape[1]} columns")

# ── Encode target ────────────────────────────────────────────────────────────
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("✔ Encoded Churn (Yes=1, No=0)")
print(f"  Class distribution — Stay: {(df['Churn']==0).sum()}  |  Churn: {(df['Churn']==1).sum()}")

# ── Encode categorical features ──────────────────────────────────────────────
cat_cols = ['gender', 'PhoneService', 'InternetService',
            'Contract', 'PaperlessBilling', 'PaymentMethod']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f"✔ One-hot encoded categoricals — {df.shape[1]} total columns")

# ── Split features and target ────────────────────────────────────────────────
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✔ Stratified split — Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

# ── Scale features ───────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print("✔ StandardScaler applied (fit on train only)")

# ── Model 1: Logistic Regression ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 1 — LOGISTIC REGRESSION")
print("=" * 60)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_s, y_train)
y_pred_lr   = lr.predict(X_test_s)
y_prob_lr   = lr.predict_proba(X_test_s)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Stay', 'Churn']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr, X_train_s, y_train, cv=cv, scoring='roc_auc')
print(f"\n5-Fold CV ROC-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Model 2: Random Forest ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 2 — RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=100, class_weight='balanced',
    random_state=42, n_jobs=-1
)
rf.fit(X_train_s, y_train)
y_pred_rf = rf.predict(X_test_s)
y_prob_rf  = rf.predict_proba(X_test_s)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Stay', 'Churn']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

cv_scores_rf = cross_val_score(rf, X_train_s, y_train, cv=cv, scoring='roc_auc')
print(f"\n5-Fold CV ROC-AUC : {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

# ── Compare and declare winner ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<25} {'Test Accuracy':>14} {'Test ROC-AUC':>13} {'CV ROC-AUC':>11}")
print("-" * 65)
print(f"{'Logistic Regression':<25} {accuracy_score(y_test, y_pred_lr):>14.4f} {roc_auc_score(y_test, y_prob_lr):>13.4f} {cv_scores.mean():>11.4f}")
print(f"{'Random Forest':<25} {accuracy_score(y_test, y_pred_rf):>14.4f} {roc_auc_score(y_test, y_prob_rf):>13.4f} {cv_scores_rf.mean():>11.4f}")

best = "Random Forest" if cv_scores_rf.mean() > cv_scores.mean() else "Logistic Regression"
print(f"\n✅ Best model by CV ROC-AUC : {best}")