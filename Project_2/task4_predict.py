import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── Load cleaned data ────────────────────────────────────────────────────────
df = pd.read_csv(r"..\DataSet\cleaned_data.csv")
print(f"✔ Loaded cleaned data — {df.shape[0]} rows")

# ── Encode target ────────────────────────────────────────────────────────────
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ── Encode categoricals (same as Task 3) ────────────────────────────────────
cat_cols = ['gender', 'PhoneService', 'InternetService',
            'Contract', 'PaperlessBilling', 'PaymentMethod']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop(columns=['Churn'])
y = df['Churn']

# ── Train on FULL dataset for prediction ─────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_scaled, y)
print("✔ Logistic Regression trained on full cleaned dataset")

# ── Store feature names for input construction ───────────────────────────────
feature_cols = X.columns.tolist()

# ── Build a sample input row using known valid values ────────────────────────
def build_input_row(tenure, monthly, total, senior, contract, internet,
                     gender, phone, paperless, payment):
    """
    Map raw user inputs to the same one-hot columns used during training.
    Returns a single-row DataFrame aligned to feature_cols.
    """
    raw = {
        'SeniorCitizen'  : senior,
        'tenure'         : tenure,
        'MonthlyCharges' : monthly,
        'TotalCharges'   : total,
    }
    # categorical fields
    cat_input = {
        'gender'          : gender,
        'PhoneService'    : phone,
        'InternetService' : internet,
        'Contract'        : contract,
        'PaperlessBilling': paperless,
        'PaymentMethod'   : payment,
    }
    # encode categoricals using get_dummies on a single-row df
    cat_df = pd.DataFrame([cat_input])
    cat_encoded = pd.get_dummies(cat_df, drop_first=True)

    row = pd.DataFrame([raw])
    row = pd.concat([row, cat_encoded], axis=1)

    # align to training columns (fill any missing dummies with 0)
    row = row.reindex(columns=feature_cols, fill_value=0)
    return row

# ── User input ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)
print("Enter customer details below.\n")

try:
    tenure   = float(input("Tenure (months)                          : "))
    monthly  = float(input("Monthly Charges ($)                      : "))
    total    = float(input("Total Charges ($)                        : "))
    senior   = int(input(  "Senior Citizen? (1=Yes / 0=No)           : "))

    print("\nContract type:")
    print("  0 = Month-to-month  |  1 = One year  |  2 = Two year")
    c_map    = {0: "Month-to-month", 1: "One year", 2: "Two year"}
    contract = c_map[int(input("Choice: "))]

    print("\nInternet Service:")
    print("  0 = DSL  |  1 = Fiber optic  |  2 = No")
    i_map    = {0: "DSL", 1: "Fiber optic", 2: "No"}
    internet = i_map[int(input("Choice: "))]

    print("\nGender  (0=Male / 1=Female): ", end="")
    gender   = "Female" if int(input()) == 1 else "Male"

    print("Phone Service    (1=Yes / 0=No): ", end="")
    phone    = "Yes" if int(input()) == 1 else "No"

    print("Paperless Billing(1=Yes / 0=No): ", end="")
    paperless = "Yes" if int(input()) == 1 else "No"

    print("\nPayment Method:")
    print("  0 = Bank transfer  |  1 = Credit card")
    print("  2 = Electronic check  |  3 = Mailed check")
    p_map    = {
        0: "Bank transfer", 1: "Credit card",
        2: "Electronic check", 3: "Mailed check"
    }
    payment  = p_map[int(input("Choice: "))]

except (ValueError, KeyError):
    print("\n❌ Invalid input. Please enter the values shown.")
    exit()

# ── Predict ──────────────────────────────────────────────────────────────────
input_row    = build_input_row(tenure, monthly, total, senior,
                                contract, internet, gender, phone, paperless, payment)
input_scaled = scaler.transform(input_row)
prediction   = model.predict(input_scaled)[0]
probability  = model.predict_proba(input_scaled)[0]

churn_prob = probability[1] * 100
stay_prob  = probability[0] * 100

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)
print(f"  Stay probability  : {stay_prob:.1f}%")
print(f"  Churn probability : {churn_prob:.1f}%")
print("-" * 60)
if prediction == 1:
    print("  ⚠  This customer is likely to CHURN.")
    if churn_prob >= 75:
        print("  Risk level: HIGH — immediate retention action recommended.")
    else:
        print("  Risk level: MODERATE — consider a targeted retention offer.")
else:
    print("  ✅ This customer is likely to STAY.")
    if stay_prob >= 80:
        print("  Risk level: LOW — no immediate action needed.")
    else:
        print("  Risk level: LOW-MEDIUM — worth monitoring.")
print("=" * 60)