import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'advertising.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Display basic statistics of the dataset to understand its distribution
print(data.describe())

# Scatter plots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(data=data, x='TV', y='Sales', ax=axs[0])
sns.scatterplot(data=data, x='Radio', y='Sales', ax=axs[1])
sns.scatterplot(data=data, x='Newspaper', y='Sales', ax=axs[2])

axs[0].set_title('TV vs Sales')
axs[1].set_title('Radio vs Sales')
axs[2].set_title('Newspaper vs Sales')

plt.show()

# Correlation matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Split the data into training and testing sets
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
