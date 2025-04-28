import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "CMAPSSData/train_FD001.txt"
data = pd.read_csv(file_path, sep=" ", header=None, engine='python')

data = data.dropna(axis=1, how='all')

# Define column names dynamically based on actual data shape
columns = ["unit", "time", "operational_setting_1", "operational_setting_2", "operational_setting_3"] + \
          [f"sensor_{i}" for i in range(1, data.shape[1] - 5 + 1)]

data.columns = columns

# Basic Data Exploration
print("Dataset Shape:", data.shape)
print("First 5 Rows:", data.head())
print("Missing Values:", data.isnull().sum())

# Visualizing Sensor Readings Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x=data["time"], y=data["sensor_2"], hue=data["unit"], legend=None, alpha=0.5)
plt.xlabel("Time Cycles")
plt.ylabel("Sensor 2 Readings")
plt.title("Sensor 2 Readings Over Time for Different Engines")
plt.show()

# Compute Remaining Useful Life (RUL)
rul = data.groupby("unit")["time"].max().reset_index()
rul.columns = ["unit", "max_time"]
data = data.merge(rul, on="unit", how="left")
data["RUL"] = data["max_time"] - data["time"]
data.drop(columns=["max_time"], inplace=True)

# Feature Correlation Analysis
correlation_matrix = data.drop(columns=["unit", "time", "RUL"]).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Matrix")
plt.show()

# Identify features with low variance
low_variance_features = [col for col in data.columns if data[col].std() < 0.01]
data = data.drop(columns=low_variance_features)
print("Dropped low variance features:", low_variance_features)

data.to_csv("processed_turbofan_data.csv", index=False)
print("\nProcessed dataset with feature selection saved as 'processed_turbofan_data.csv'")