# analyze_and_visualize.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set Seaborn style for better visuals
sns.set(style="whitegrid")

# ----------------------------
# Task 1: Load and Explore the Dataset
# ----------------------------

try:
    # Load Iris dataset from sklearn and convert to pandas DataFrame
    iris_raw = load_iris()
    iris_df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)

    print("First 5 rows of the dataset:")
    print(iris_df.head())

    print("\nDataset info:")
    print(iris_df.info())

    print("\nMissing values check:")
    print(iris_df.isnull().sum())

    # No missing values, so no cleaning needed

except Exception as e:
    print(f"Error loading dataset: {e}")

# ----------------------------
# Task 2: Basic Data Analysis
# ----------------------------

print("\nBasic Statistics:")
print(iris_df.describe())

print("\nMean values by species:")
grouped_means = iris_df.groupby('species').mean()
print(grouped_means)

# Optional: Print any interesting pattern
print("\nInteresting Finding: On average, 'virginica' species has the highest petal length and width.")

# ----------------------------
# Task 3: Data Visualization
# ----------------------------

# 1. Line Chart (simulate time series using index for simplicity)
plt.figure(figsize=(8, 5))
plt.plot(iris_df.index, iris_df["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart of Sepal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("line_chart.png")
plt.show()

# 2. Bar Chart (average petal length per species)
plt.figure(figsize=(6, 4))
sns.barplot(data=iris_df, x="species", y="petal length (cm)")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.show()a

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(6, 4))
plt.hist(iris_df["sepal width (cm)"], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram.png")
plt.show()

# 4. Scatter Plot (sepal length vs petal length)
plt.figure(figsize=(6, 5))
sns.scatterplot(data=iris_df, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

# ----------------------------
# Observations
# ----------------------------
print("\nObservations:")
print("- 'Setosa' species has significantly smaller petal lengths and widths.")
print("- 'Virginica' species tends to have the largest features across the board.")
print("- There's a strong positive correlation between sepal length and petal length.")

