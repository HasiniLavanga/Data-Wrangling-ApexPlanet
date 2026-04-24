# ============================================
# TASK-1: DATA IMMERSION & WRANGLING (FINAL)
# ============================================

import pandas as pd
import numpy as np

# --------------------------------------------
# 1. LOAD DATA
# --------------------------------------------
print("📂 Loading dataset...")

df = pd.read_csv("SampleSuperstore.csv", encoding='latin1')

print("\n✅ First 5 rows:")
print(df.head())

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe())


# --------------------------------------------
# 2. DATA PROFILING
# --------------------------------------------
print("\n🔍 Checking Missing Values:")
print(df.isnull().sum())

print("\n🔁 Checking Duplicate Records:")
print("Duplicate rows:", df.duplicated().sum())


# --------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------
print("\n🧹 Cleaning Data...")

df = df.drop_duplicates()

for col in ['city', 'state']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.title().str.strip()

if 'quantity' in df.columns:
    df = df[df['quantity'] > 0]

if 'sales' in df.columns:
    df = df[df['sales'] >= 0]

print("✅ Cleaning Completed!")


# --------------------------------------------
# 4. OUTLIER HANDLING
# --------------------------------------------
print("\n📉 Handling Outliers...")

for col in ['sales', 'profit', 'quantity']:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.clip(df[col], lower, upper)

print("✅ Outliers handled!")


# --------------------------------------------
# 5. FEATURE ENGINEERING
# --------------------------------------------
print("\n⚙️ Feature Engineering...")

df['profit_ratio'] = np.where(df['sales'] != 0,
                             df['profit'] / df['sales'],
                             0)

df['sales_category'] = pd.qcut(
    df['sales'],
    q=5,
    labels=['Low', 'Medium', 'High', 'Very High', 'Premium']
)

print("✅ Feature Engineering Done!")


# --------------------------------------------
# 6. FINAL DATA CHECK
# --------------------------------------------
print("\n📊 Final Shape:", df.shape)


# --------------------------------------------
# 7. SAVE CLEAN DATASET (ERROR-PROOF)
# --------------------------------------------
print("\n💾 Saving Clean Dataset...")

try:
    df.to_csv("Cleaned_Superstore.csv", index=False)
    print("✅ Saved as Cleaned_Superstore.csv")
except PermissionError:
    df.to_csv("Cleaned_Superstore_new.csv", index=False)
    print("⚠️ File open, saved as Cleaned_Superstore_new.csv")


# --------------------------------------------
# 8. CREATE DATA DICTIONARY (ERROR-PROOF)
# --------------------------------------------
print("\n📘 Creating Data Dictionary...")

data_dict = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Description": ["Add description here"] * len(df.columns)
})

try:
    data_dict.to_csv("Data_Dictionary.csv", index=False)
    print("✅ Data Dictionary saved")
except PermissionError:
    data_dict.to_csv("Data_Dictionary_new.csv", index=False)
    print("⚠️ File open, saved as Data_Dictionary_new.csv")


# --------------------------------------------
# 9. SAMPLE INSIGHTS
# --------------------------------------------
print("\n📊 Sample Insights:")

print("\nTop Sales by Region:")
print(df.groupby('region')['sales'].sum().sort_values(ascending=False))

print("\nProfit by Category:")
print(df.groupby('category')['profit'].sum().sort_values(ascending=False))


print("\n🎯 TASK COMPLETED SUCCESSFULLY!")
