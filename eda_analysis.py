# ============================================
# TASK-2: EDA & BUSINESS INTELLIGENCE
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# 1. LOAD DATA
# --------------------------------------------
print("📂 Loading cleaned dataset...")

df = pd.read_csv("Cleaned_Superstore.csv")

print("\n✅ First 5 rows:")
print(df.head())


# --------------------------------------------
# 2. SUMMARY STATISTICS
# --------------------------------------------
print("\n📊 Statistical Summary:")
print(df.describe())

print("\n📊 Category Distribution:")
print(df['category'].value_counts())


# --------------------------------------------
# 3. UNIVARIATE ANALYSIS
# --------------------------------------------
print("\n📈 Creating Univariate Plots...")

# Sales Distribution
plt.figure()
df['sales'].hist()
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.savefig("sales_distribution.png")
plt.show()

# Profit Distribution
plt.figure()
df['profit'].hist()
plt.title("Profit Distribution")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.savefig("profit_distribution.png")
plt.show()

# Category Count
plt.figure()
df['category'].value_counts().plot(kind='bar')
plt.title("Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.savefig("category_distribution.png")
plt.show()


# --------------------------------------------
# 4. BIVARIATE ANALYSIS
# --------------------------------------------
print("\n📊 Creating Bivariate Plots...")

# Sales by Region
plt.figure()
df.groupby('region')['sales'].sum().plot(kind='bar')
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.savefig("sales_by_region.png")
plt.show()

# Profit by Category
plt.figure()
df.groupby('category')['profit'].sum().plot(kind='bar')
plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Total Profit")
plt.savefig("profit_by_category.png")
plt.show()

# Scatter Plot
plt.figure()
plt.scatter(df['sales'], df['profit'])
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.title("Sales vs Profit")
plt.savefig("sales_vs_profit.png")
plt.show()


# --------------------------------------------
# 5. MULTIVARIATE ANALYSIS
# --------------------------------------------
print("\n🔥 Creating Advanced Visualizations...")

# Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Pairplot
sns.pairplot(df[['sales', 'profit', 'quantity']])
plt.savefig("pairplot.png")
plt.show()


# --------------------------------------------
# 6. BUSINESS INSIGHTS
# --------------------------------------------
print("\n📊 BUSINESS INSIGHTS:")

print("\nTop 5 Sub-Categories by Sales:")
print(df.groupby('sub-category')['sales'].sum().sort_values(ascending=False).head())

print("\nTop 5 Cities by Sales:")
print(df.groupby('city')['sales'].sum().sort_values(ascending=False).head())

print("\nAverage Discount vs Profit:")
print(df.groupby('discount')['profit'].mean().sort_values())


# --------------------------------------------
# 7. SAVE INSIGHTS
# --------------------------------------------
print("\n💾 Saving Insights...")

insights = df.groupby('region')[['sales','profit']].sum()
insights.to_csv("eda_insights.csv")

print("✅ Insights saved as eda_insights.csv")


# --------------------------------------------
# 8. COMPLETION
# --------------------------------------------
print("\n🎯 TASK-2 EDA COMPLETED SUCCESSFULLY!")