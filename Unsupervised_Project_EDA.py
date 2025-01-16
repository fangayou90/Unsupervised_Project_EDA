
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
modis_data = pd.DataFrame({
    'Region': [f'Region_{i}' for i in range(10000)],
    'NDVI': [round(0.5 + i * 0.00001, 5) for i in range(10000)],
    'EVI': [round(0.3 + i * 0.00001, 5) for i in range(10000)],
    'Latitude': [round(10 + i * 0.0001, 5) for i in range(10000)],
    'Longitude': [round(-50 + i * 0.0001, 5) for i in range(10000)]
})

# Display the shape of the dataset
print("Dataset Shape:", modis_data.shape)

# Display the first few rows
print(modis_data.head())

# Summary statistics
print("Summary Statistics:")
print(modis_data.describe())

# Pair plot for key variables
sns.pairplot(modis_data[['NDVI', 'EVI', 'Latitude', 'Longitude']])
plt.title("Pair Plot")
plt.show()

# Correlation matrix
corr_matrix = modis_data[['NDVI', 'EVI', 'Latitude', 'Longitude']].corr()

# Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()

# NDVI distribution plot
sns.histplot(modis_data['NDVI'], kde=True)
plt.title("NDVI Distribution")
plt.xlabel("NDVI")
plt.show()

# Scatter plot of Latitude vs NDVI
sns.scatterplot(data=modis_data, x='Latitude', y='NDVI')
plt.title("Latitude vs NDVI")
plt.xlabel("Latitude")
plt.ylabel("NDVI")
plt.show()

# Empirical Cumulative Distribution Function (CDF) of NDVI
sns.ecdfplot(data=modis_data, x='NDVI')
plt.title("CDF of NDVI")
plt.xlabel("NDVI")
plt.ylabel("Proportion")
plt.show()

# Check for missing data
missing_data = modis_data.isnull().sum()
print("Missing Data:")
print(missing_data)

# Summary of Findings and Questions
print("Summary of Findings and Questions:")
print("1. The dataset has 10,000 rows and 5 columns, making it manageable for analysis in standard tools.")
print("2. The data appears to cover geographic coordinates and vegetation indices (NDVI and EVI).")
print("3. Key Questions:")
print("   - Can clusters of regions with similar vegetation patterns be identified?")
print("   - Are there noticeable geographic trends in NDVI or EVI?")
print("   - What relationships exist between NDVI and EVI?")
print("4. This dataset does not qualify as 'big data' but is well-suited for clustering tasks.")
