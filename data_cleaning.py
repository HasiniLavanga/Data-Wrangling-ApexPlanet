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

# Standardize column names (IMPORTANT)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe())

print("\n📌 Columns:")
print(df.columns)


# --------------------------------------------
# 2. DATA PROFILING
# --------------------------------------------
print("\n🔍 Checking Missing Values:")
print(df.isnull().sum())

print("\n🔁 Checking Duplicate Records:")
print("Duplicate rows:", df.duplicated().sum())

print("\n📊 Unique Values Per Column:")
print(df.nunique())


# --------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------
print("\n🧹 Cleaning Data...")

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
if 'postal_code' in df.columns:
    df['postal_code'] = df['postal_code'].fillna('unknown')

# Standardize text columns
for col in ['city', 'state']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.title().str.strip()

# Fix invalid values
if 'quantity' in df.columns:
    df = df[df['quantity'] > 0]

if 'sales' in df.columns:
    df = df[df['sales'] >= 0]

print("\n✅ Cleaning Completed!")


# --------------------------------------------
# 4. DATE FORMATTING & FEATURE EXTRACTION
# --------------------------------------------
print("\n📅 Handling Date Columns...")

for col in ['order_date', 'ship_date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Extract year and month
if 'order_date' in df.columns:
    df['order_year'] = df['order_date'].dt.year
    df['order_month'] = df['order_date'].dt.month

print("✅ Date formatting done!")


# --------------------------------------------
# 5. OUTLIER HANDLING (CLIPPING METHOD)
# --------------------------------------------
print("\n📉 Handling Outliers (Clipping)...")

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
# 6. FEATURE ENGINEERING
# --------------------------------------------
print("\n⚙️ Feature Engineering...")

# Profit Ratio (safe calculation)
if 'profit' in df.columns and 'sales' in df.columns:
    df['profit_ratio'] = np.where(df['sales'] != 0,
                                 df['profit'] / df['sales'],
                                 0)

# Sales Category
if 'sales' in df.columns:
    df['sales_category'] = pd.qcut(
        df['sales'],
        q=5,
        labels=['Low', 'Medium', 'High', 'Very High', 'Premium']
    )

print("✅ Feature Engineering Done!")


# --------------------------------------------
# 7. FINAL DATA CHECK
# --------------------------------------------
print("\n📊 Final Dataset Info:")
print(df.info())

print("\n📌 Final Shape:", df.shape)


# --------------------------------------------
# 8. SAVE CLEAN DATASET
# --------------------------------------------
print("\n💾 Saving Clean Dataset...")

df.to_csv("Cleaned_Superstore.csv", index=False)

print("✅ Cleaned dataset saved as 'Cleaned_Superstore.csv'")


# --------------------------------------------
# 9. DATA DICTIONARY CREATION
# --------------------------------------------
print("\n📘 Creating Data Dictionary...")

data_dict = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Description": [
        "Add appropriate description here"
        for _ in df.columns
    ]
})

data_dict.to_csv("Data_Dictionary.csv", index=False)

print("✅ Data Dictionary saved!")


# --------------------------------------------
# 10. SAMPLE INSIGHTS
# --------------------------------------------
print("\n📊 Sample Insights:")

if 'region' in df.columns and 'sales' in df.columns:
    print("\nTop Sales by Region:")
    print(df.groupby('region')['sales'].sum().sort_values(ascending=False))

if 'category' in df.columns and 'profit' in df.columns:
    print("\nProfit by Category:")
    print(df.groupby('category')['profit'].sum().sort_values(ascending=False))


print("\n🎯 TASK COMPLETED SUCCESSFULLY!")