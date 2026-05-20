import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, classification_report
)

# ── Load cleaned data ────────────────────────────────────────────────────────
df = pd.read_csv(r"..\DataSet\cleaned_data.csv")
print(f"✔ Loaded cleaned data — {df.shape[0]} rows, {df.shape[1]} columns")

# ── Train model (consistent with Task 3 / Task 4) ───────────────────────────
df_model = df.copy()
df_model['Churn_num'] = df_model['Churn'].map({'Yes': 1, 'No': 0})
cat_cols = ['gender', 'PhoneService', 'InternetService',
            'Contract', 'PaperlessBilling', 'PaymentMethod']
df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

X = df_encoded.drop(columns=['Churn', 'Churn_num'])
y = df_encoded['Churn_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_s, y_train)
y_pred  = model.predict(X_test_s)
y_prob  = model.predict_proba(X_test_s)[:, 1]
cm      = confusion_matrix(y_test, y_pred)
acc     = accuracy_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

report = classification_report(
    y_test, y_pred,
    target_names=['Stay', 'Churn'],
    output_dict=True
)
print(f"✔ Model ready — Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f}")

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : '#FAFAFA',
    'axes.facecolor'   : '#FFFFFF',
    'axes.edgecolor'   : '#E0E0E0',
    'axes.grid'        : True,
    'grid.color'       : '#F0F0F0',
    'grid.linestyle'   : '-',
    'grid.linewidth'   : 0.8,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.spines.left' : False,
    'axes.spines.bottom': True,
    'xtick.color'      : '#888888',
    'ytick.color'      : '#888888',
    'axes.labelcolor'  : '#444444',
    'axes.titlesize'   : 13,
    'axes.titleweight' : '500',
    'axes.titlepad'    : 12,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'font.family'      : 'DejaVu Sans',
    'text.color'       : '#333333',
})

BLUE  = '#185FA5'
RED   = '#D85A30'
TEAL  = '#1D9E75'
AMBER = '#BA7517'
GRAY  = '#888780'

def note(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes,
            fontsize=10, color='#888888', va='bottom')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Dataset overview
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(13, 9))
fig1.patch.set_facecolor('#FAFAFA')
fig1.suptitle('ChurnGuard — dataset overview', fontsize=16,
              fontweight='500', color='#222222', y=1.01)

# 1a: Churn distribution
ax = axes[0, 0]
counts = df['Churn'].value_counts()
bars   = ax.bar(counts.index, counts.values, color=[BLUE, RED], width=0.5, zorder=3)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='500')
ax.set_title('Churn distribution')
note(ax, f'Total customers: {len(df):,}  |  Churn rate: {counts.get("Yes",0)/len(df)*100:.1f}%')
ax.set_ylabel('Customers')
ax.set_ylim(0, max(counts.values) * 1.18)
ax.tick_params(bottom=False)

# 1b: Churn by contract type
ax = axes[0, 1]
ct  = df.groupby(['Contract', 'Churn']).size().unstack(fill_value=0)
x   = np.arange(len(ct)); w = 0.35
b1  = ax.bar(x - w/2, ct['No'],  width=w, color=BLUE, label='Stay',  zorder=3)
b2  = ax.bar(x + w/2, ct['Yes'], width=w, color=RED,  label='Churn', zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(ct.index, fontsize=9)
ax.set_title('Churn by contract type')
note(ax, 'Month-to-month customers churn most')
ax.set_ylabel('Customers')
ax.legend(fontsize=9, frameon=False)
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 3, str(int(h)),
                ha='center', va='bottom', fontsize=8, color='#666666')

# 1c: Churn rate by internet service (stacked %)
ax = axes[1, 0]
ci     = df.groupby(['InternetService', 'Churn']).size().unstack(fill_value=0)
ci_pct = ci.div(ci.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(ci_pct))
for col, color, lbl in [('No', BLUE, 'Stay'), ('Yes', RED, 'Churn')]:
    vals = ci_pct[col].values
    bars = ax.bar(ci_pct.index, vals, bottom=bottom, color=color, label=lbl, zorder=3)
    for bar, v, b in zip(bars, vals, bottom):
        if v > 5:
            ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2,
                    f'{v:.0f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='500')
    bottom += vals
ax.set_title('Churn rate by internet service')
note(ax, '% of each service group that churned')
ax.set_ylabel('Percentage (%)')
ax.set_ylim(0, 115)
ax.legend(fontsize=9, frameon=False)

# 1d: Monthly charges histogram by churn
ax = axes[1, 1]
stay_mc  = df[df['Churn'] == 'No']['MonthlyCharges']
churn_mc = df[df['Churn'] == 'Yes']['MonthlyCharges']
ax.hist(stay_mc,  bins=25, color=BLUE, alpha=0.7,
        label=f'Stay  (mean=${stay_mc.mean():.0f})',  zorder=3)
ax.hist(churn_mc, bins=25, color=RED,  alpha=0.7,
        label=f'Churn (mean=${churn_mc.mean():.0f})', zorder=3)
ax.set_title('Monthly charges distribution')
note(ax, 'Churners tend to pay more per month')
ax.set_xlabel('Monthly charges ($)')
ax.set_ylabel('Count')
ax.legend(fontsize=9, frameon=False)

plt.tight_layout(pad=2.5)
plt.savefig('task5_fig1_overview.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print('✔ Figure 1 saved — task5_fig1_overview.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Customer behaviour deep-dive
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 2, figsize=(13, 9))
fig2.patch.set_facecolor('#FAFAFA')
fig2.suptitle('ChurnGuard — customer behaviour deep-dive',
              fontsize=16, fontweight='500', color='#222222', y=1.01)

# 2a: Tenure vs Monthly charges scatter
ax = axes[0, 0]
for label, color in [('No', BLUE), ('Yes', RED)]:
    sub = df[df['Churn'] == label]
    ax.scatter(sub['tenure'], sub['MonthlyCharges'], alpha=0.25, s=18,
               color=color, label='Stay' if label == 'No' else 'Churn', zorder=3)
ax.set_title('Tenure vs monthly charges')
note(ax, 'Coloured by churn outcome')
ax.set_xlabel('Tenure (months)')
ax.set_ylabel('Monthly charges ($)')
ax.legend(fontsize=9, frameon=False, markerscale=1.5)

# 2b: Senior vs Non-senior churn rate
ax = axes[0, 1]
cs     = df.groupby(['SeniorCitizen', 'Churn']).size().unstack(fill_value=0)
cs_pct = cs.div(cs.sum(axis=1), axis=0) * 100
labels = ['Non-senior', 'Senior']
x = np.arange(2); w = 0.35
ax.bar(x - w/2, cs_pct['No'],  width=w, color=BLUE, label='Stay',  zorder=3)
ax.bar(x + w/2, cs_pct['Yes'], width=w, color=RED,  label='Churn', zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title('Churn rate — senior vs non-senior')
note(ax, 'Senior citizens churn at higher rates')
ax.set_ylabel('Percentage (%)')
ax.legend(fontsize=9, frameon=False)
for xi, (ns, ys) in enumerate(zip(cs_pct['No'], cs_pct['Yes'])):
    ax.text(xi - w/2, ns + 1, f'{ns:.0f}%', ha='center', va='bottom', fontsize=9)
    ax.text(xi + w/2, ys + 1, f'{ys:.0f}%', ha='center', va='bottom', fontsize=9)

# 2c: Churn rate by payment method
ax = axes[1, 0]
pm     = df.groupby(['PaymentMethod', 'Churn']).size().unstack(fill_value=0)
pm_pct = (pm['Yes'] / pm.sum(axis=1) * 100).sort_values(ascending=True)
colors_pm = [TEAL if v < 33 else AMBER if v < 40 else RED for v in pm_pct.values]
bars = ax.barh(pm_pct.index, pm_pct.values, color=colors_pm, zorder=3, height=0.5)
for bar, val in zip(bars, pm_pct.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', fontsize=9)
ax.set_title('Churn rate by payment method')
note(ax, '% of each payment group that churned')
ax.set_xlabel('Churn rate (%)')
ax.set_xlim(0, 55)
ax.tick_params(left=False)

# 2d: Correlation heatmap (numeric features)
ax = axes[1, 1]
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
df_corr  = df[num_cols + ['Churn']].copy()
df_corr['Churn'] = df_corr['Churn'].map({'Yes': 1, 'No': 0})
corr_matrix = df_corr.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, cbar=True, linewidths=0.5,
            annot_kws={'size': 10},
            xticklabels=['Tenure', 'Monthly\nCharges', 'Total\nCharges', 'Senior', 'Churn'],
            yticklabels=['Tenure', 'Monthly\nCharges', 'Total\nCharges', 'Senior', 'Churn'])
ax.set_title('Correlation heatmap (numerical features)')
note(ax, 'Churn negatively correlated with tenure')
ax.tick_params(left=False, bottom=False)

plt.tight_layout(pad=2.5)
plt.savefig('task5_fig2_behaviour.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print('✔ Figure 2 saved — task5_fig2_behaviour.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Model evaluation
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
fig3.patch.set_facecolor('#FAFAFA')
fig3.suptitle(
    f'ChurnGuard — model evaluation  |  Logistic Regression (class_weight=balanced)  |  '
    f'Accuracy: {acc:.1%}  |  ROC-AUC: {auc:.3f}',
    fontsize=12, fontweight='500', color='#222222', y=1.03
)

# 3a: Confusion matrix
ax = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'],
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 14, 'weight': '500'},
            ax=ax, cbar=False)
ax.set_title('Confusion matrix')
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
ax.tick_params(left=False, bottom=False)

# 3b: Precision / Recall / F1 per class
ax = axes[1]
categories = ['Stay', 'Churn']
precision  = [report['Stay']['precision'], report['Churn']['precision']]
recall     = [report['Stay']['recall'],    report['Churn']['recall']]
f1         = [report['Stay']['f1-score'],  report['Churn']['f1-score']]
x = np.arange(2); w = 0.22
ax.bar(x - w,   precision, width=w, color=BLUE,  label='Precision', zorder=3)
ax.bar(x,       recall,    width=w, color=TEAL,  label='Recall',    zorder=3)
ax.bar(x + w,   f1,        width=w, color=AMBER, label='F1-score',  zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 1.15)
ax.set_title('Precision · recall · F1 by class')
note(ax, f'Overall accuracy: {acc:.1%}')
ax.set_ylabel('Score')
ax.legend(fontsize=9, frameon=False)
for xi in range(2):
    for offset, val, color in [(-w, precision[xi], BLUE), (0, recall[xi], TEAL), (w, f1[xi], AMBER)]:
        ax.text(xi + offset, val + 0.02, f'{val:.2f}',
                ha='center', va='bottom', fontsize=8, color=color)

# 3c: ROC curve
ax = axes[2]
ax.plot(fpr, tpr, color=BLUE, lw=2, label=f'ROC curve (AUC = {auc:.3f})', zorder=3)
ax.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle='--', label='Random baseline', zorder=2)
ax.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
ax.set_title('ROC curve')
note(ax, 'Higher AUC = better churn detection')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.legend(fontsize=9, frameon=False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

plt.tight_layout(pad=2.5)
plt.savefig('task5_fig3_model.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print('✔ Figure 3 saved — task5_fig3_model.png')

print('\n✅ All 3 figures saved successfully.')