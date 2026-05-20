import pandas as pd

df = pd.read_csv(r"..\DataSet\churnguard_data.csv")
df.columns = df.columns.str.strip()

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head().to_string())

print("\n" + "=" * 60)
print("COLUMN NAMES AND DATA TYPES (.info())")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY (.describe())")
print("=" * 60)
print(df.describe().to_string())

print("\n" + "=" * 60)
print("MISSING VALUES PER COLUMN")
print("=" * 60)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values found.")
else:
    print(missing[missing > 0].to_string())
print(f"\nTotal missing cells : {missing.sum()}")

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(f"Number of duplicate rows : {df.duplicated().sum()}")

print("\n" + "=" * 60)
print("CHURN COLUMN — VALUE COUNTS (inconsistent casing expected)")
print("=" * 60)
churn_counts = df['Churn'].value_counts(dropna=False)
print(churn_counts.to_string())
total = len(df)
yes_approx = churn_counts[churn_counts.index.str.lower().isin(['yes'])].sum()
print(f"\nApprox churn rate : {yes_approx / total * 100:.1f}%")

print("\n" + "=" * 60)
print("CONTRACT COLUMN — UNIQUE VALUES (typos/variants expected)")
print("=" * 60)
for val in sorted(df['Contract'].unique(), key=lambda x: str(x)):
    print(f"  '{val}'")

print("\n" + "=" * 60)
print("INTERNET SERVICE — UNIQUE VALUES")
print("=" * 60)
for val in sorted(df['InternetService'].dropna().unique(), key=lambda x: str(x)):
    print(f"  '{val}'")

print("\n" + "=" * 60)
print("PAYMENT METHOD — UNIQUE VALUES (whitespace variants expected)")
print("=" * 60)
for val in df['PaymentMethod'].unique():
    print(f"  '{val}'")

print("\n" + "=" * 60)
print("CATEGORICAL COLUMNS — VALUE COUNTS")
print("=" * 60)
cat_cols = ['gender', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).to_string())

print("\n" + "=" * 60)
print("OBSERVATIONS SUMMARY")
print("=" * 60)
print("  1. Churn column has mixed casing (Yes/YES/yes/No/NO/no etc.)")
print("  2. Contract column has typos and alternate formats")
print("  3. InternetService has spelling variants (FiberOptic, Fibre optic etc.)")
print("  4. PaymentMethod has leading whitespace in some entries")
print("  5. TotalCharges is stored as object — needs numeric conversion")
print("  6. tenure and MonthlyCharges have missing values")
print("  7. MonthlyCharges has outlier (max=1500, normal range ~$20-$120)")
print("  8. tenure has negative values which are invalid")