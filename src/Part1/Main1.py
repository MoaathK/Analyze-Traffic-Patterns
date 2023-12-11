
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Read the CSV file
csv_file_path = r"Traffic.csv"
traffic_data = pd.read_csv(csv_file_path)

# Create subplots for histograms
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Histograms of Vehicle Counts', fontsize=16)

sns.histplot(traffic_data['CarCount'], bins=30, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Car Count')

sns.histplot(traffic_data['BikeCount'], bins=30, kde=True, ax=axes[0, 1], color='green')
axes[0, 1].set_title('Bike Count')

sns.histplot(traffic_data['BusCount'], bins=30, kde=True, ax=axes[1, 0], color='orange')
axes[1, 0].set_title('Bus Count')

sns.histplot(traffic_data['TruckCount'], bins=30, kde=True, ax=axes[1, 1], color='red')
axes[1, 1].set_title('Truck Count')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Create boxplot
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=traffic_data[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']], palette="Set2")
ax.set_title('Boxplot of Vehicle Counts')

# Create subplots for scatter plots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Scatter Plots of Vehicle Counts vs Total', fontsize=16)

sns.scatterplot(data=traffic_data, x='Total', y='CarCount', ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Car Count vs Total')

sns.scatterplot(data=traffic_data, x='Total', y='BikeCount', ax=axes[0, 1], color='green')
axes[0, 1].set_title('Bike Count vs Total')

sns.scatterplot(data=traffic_data, x='Total', y='BusCount', ax=axes[1, 0], color='orange')
axes[1, 0].set_title('Bus Count vs Total')

sns.scatterplot(data=traffic_data, x='Total', y='TruckCount', ax=axes[1, 1], color='red')
axes[1, 1].set_title('Truck Count vs Total')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Compute the correlation matrix
correlation_matrix = traffic_data[['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']].corr()

# Create a heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Vehicle Counts')

plt.show()