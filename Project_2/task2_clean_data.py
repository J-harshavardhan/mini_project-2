import pandas as pd

df = pd.read_csv(r"..\DataSet\churnguard_data.csv")
df.columns = df.columns.str.strip()
print(f"Original shape : {df.shape}")

# ── Step 1: Drop customerID (not a feature) ─────────────────────────────────
df.drop(columns=["customerID"], inplace=True)
print("✔ Dropped customerID column")

# ── Step 2: Remove duplicates ────────────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"✔ Removed {before - len(df)} duplicate rows")

# ── Step 3: Strip whitespace from all string columns ────────────────────────
str_cols = df.select_dtypes(include="object").columns
for col in str_cols:
    df[col] = df[col].str.strip()
print("✔ Stripped leading/trailing whitespace from all string columns")

# ── Step 4: Standardise Churn, PhoneService, PaperlessBilling ───────────────
for col in ['Churn', 'PhoneService', 'PaperlessBilling']:
    df[col] = df[col].str.title()
print("✔ Standardised casing → Churn, PhoneService, PaperlessBilling")

# ── Step 5: Fix gender ───────────────────────────────────────────────────────
df['gender'] = df['gender'].str.title()
print("✔ Standardised gender casing")

# ── Step 6: Fix Contract variants ───────────────────────────────────────────
def fix_contract(val):
    if pd.isna(val):
        return val
    v = str(val).strip().lower()
    if v in ("month-to-month", "month to month", "monthly"):
        return "Month-to-month"
    elif v in ("one year", "one-year", "1 year", "1-year"):
        return "One year"
    elif v in ("two year", "two-year", "2 year", "2-year"):
        return "Two year"
    return None

df["Contract"] = df["Contract"].apply(fix_contract)
print("✔ Fixed Contract values → Month-to-month / One year / Two year")

# ── Step 7: Fix InternetService variants ────────────────────────────────────
def fix_internet(val):
    if pd.isna(val):
        return val
    v = str(val).strip().lower().replace(" ", "").replace("_", "")
    if v == "dsl":
        return "DSL"
    elif v in ("fiberoptic", "fibreoptic"):
        return "Fiber optic"
    elif v in ("no", "none", "nan"):
        return "No"
    return None

df["InternetService"] = df["InternetService"].apply(fix_internet)
print("✔ Fixed InternetService values → DSL / Fiber optic / No")

# ── Step 8: Drop rows with unrecognised Contract or InternetService ──────────
before = len(df)
df.dropna(subset=["Contract", "InternetService"], inplace=True)
print(f"✔ Removed {before - len(df)} rows with unrecognisable Contract/InternetService")

# ── Step 9: Convert TotalCharges to numeric ──────────────────────────────────
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("✔ Converted TotalCharges to numeric (non-numeric entries → NaN)")

# ── Step 10: Remove invalid tenure (zero or negative) ───────────────────────
before = len(df)
df = df[~(df['tenure'].notna() & (df['tenure'] <= 0))]
print(f"✔ Removed {before - len(df)} rows with zero/negative tenure")

# ── Step 11: Remove MonthlyCharges outliers ──────────────────────────────────
before = len(df)
df = df[~(df['MonthlyCharges'].notna() & ((df['MonthlyCharges'] < 10) | (df['MonthlyCharges'] > 200)))]
print(f"✔ Removed {before - len(df)} rows with outlier MonthlyCharges (< $10 or > $200)")

# ── Step 12: Impute missing values ───────────────────────────────────────────
mc_mean  = df['MonthlyCharges'].mean()
tc_mean  = df['TotalCharges'].mean()
ten_med  = int(round(df['tenure'].median()))

df['MonthlyCharges'] = df['MonthlyCharges'].fillna(mc_mean)
df['TotalCharges']   = df['TotalCharges'].fillna(tc_mean)
df['tenure']         = df['tenure'].fillna(ten_med)

print(f"✔ Imputed MonthlyCharges  NaNs with mean   (${mc_mean:.2f})")
print(f"✔ Imputed TotalCharges    NaNs with mean   (${tc_mean:.2f})")
print(f"✔ Imputed tenure          NaNs with median ({ten_med} months)")

# ── Step 13: Save cleaned dataset ────────────────────────────────────────────
df.to_csv(r"..\DataSet\cleaned_data.csv", index=False)
print("✔ Saved cleaned dataset → cleaned_data.csv")

# ── Final report ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLEANED DATASET — SHAPE")
print("=" * 60)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\n" + "=" * 60)
print("MISSING VALUE CHECK AFTER CLEANING")
print("=" * 60)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ No missing values remaining.")
else:
    print(missing[missing > 0].to_string())

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION (CLEANED)")
print("=" * 60)
churn_counts = df['Churn'].value_counts()
print(churn_counts.to_string())
print(f"\nChurn rate : {churn_counts.get('Yes', 0) / len(df) * 100:.1f}%")

print("\n" + "=" * 60)
print("SAMPLE OF CLEANED DATA (first 5 rows)")
print("=" * 60)
print(df.head().to_string())