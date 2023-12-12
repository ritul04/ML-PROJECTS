#!/usr/bin/env python
# coding: utf-8

# In[231]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Downloaded datasets from FRED

# In[232]:


import pandas as pd

# Read the CSV files
cpi = pd.read_csv('cpi.csv')
job_growth = pd.read_csv('Jobgrowth.csv')
mortgage = pd.read_csv('mortgage.csv')
money_supply = pd.read_csv('msupply.csv')
population = pd.read_csv('population.csv')
unemployment_rate = pd.read_csv('unrate.csv')
price = pd.read_csv('price.csv')


# In[233]:


# Display the first few rows of each dataset
print("CPI Dataset:")
print(cpi.head())

print("\nJob Growth Dataset:")
print(job_growth.head())

print("\nMortgage Dataset:")
print(mortgage.head())

print("\nMoney Supply Dataset:")
print(money_supply.head())

print("\nPopulation Dataset:")
print(population.head())

print("\nUnemployment Rate Dataset:")
print(unemployment_rate.head())

print("\nHouse Price Dataset:")
print(price.head())


# In[234]:


import pandas as pd
import glob

# Getting a list of all CSV files in the current directory
csv_files = glob.glob('*.csv')

# Loop through each CSV file and display its null values
for file in csv_files:
    df = pd.read_csv(file)
    print(f"null values {file}")
    print(df.isnull().sum())  
    print("\n")  # Adding a line break between file null value count


# In[235]:


# Merge datasets based on the 'DATE' column
merged_data = pd.merge(cpi, job_growth, on='DATE', how='inner')
merged_data = pd.merge(merged_data, mortgage, on='DATE', how='inner')
merged_data = pd.merge(merged_data, money_supply, on='DATE', how='inner')
merged_data = pd.merge(merged_data, population, on='DATE', how='inner')
merged_data = pd.merge(merged_data, unemployment_rate, on='DATE', how='inner')
merged_data = pd.merge(merged_data, price, on='DATE', how='inner')
# Check the shape and sample of the combined dataset
print(merged_data.shape)
print(merged_data.head())


# In[236]:


merged_data.info()


# In[237]:


merged_data.to_csv('D:excel/merged_data.csv', index=False)


# In[238]:


price


# In[239]:



# Display summary statistics of numerical columns
merged_data.describe()


# In[240]:


# Check for missing values
missing_values = merged_data.isnull().sum()
print(missing_values)


# In[241]:


# Plot histograms for numerical columns
import matplotlib.pyplot as plt

numerical_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(12, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, len(numerical_columns) // 2 + 1, i)
    plt.hist(merged_data[column], bins=20)
    plt.title(column)
    plt.xlabel('Values')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[242]:


# Plot correlation heatmap
correlation_matrix = merged_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# Interpreting the Heatmap:
# A correlation coefficient ranges between -1 to 1, where:
# 1 indicates a perfect positive correlation (as one variable increases, the other also increases proportionally).
# -1 indicates a perfect negative correlation (as one variable increases, the other decreases proportionally).
# 0 indicates no correlation.
# Interpretations from the Heatmap:
#     
# #Strong positive correlation:
# 
# CPI and POPULATION have a strong positive correlation of 0.95. This suggests that as one variable increases, the other tends to increase almost in lockstep.
# 
# CPI and PRICE also show a relatively strong positive correlation of 0.83.
# 
# #Moderate positive correlation:
# 
# PRICE and POPULATION exhibit a moderate positive correlation of 0.68.
# 
# JOB GROWTH and PRICE have a moderate positive correlation of 0.64.
# 
# MORTGAGE and JOB GROWTH have a moderate positive correlation of 0.59.
# 
# #Weak positive correlation:
# 
# UNRATE and CPI show a weak positive correlation of 0.39.
# 
# JOB GROTH and PRICE show a weak positive correlation of 0.25
# 
# #Weak negative correlation:
# 
# MORTGAGE and POPULATION have a weak negative correlation of -0.63.
# 
# UNRATE and JOB GROWTH have a weak negative correlation of -0.55.
# 
# 
# UNRATE and PRICE have a weak negative correlation of -0.54.
# 
# CPI and MORTGAGE have a weak negative correlation of -0.42.

# In[243]:


# Boxplots for numerical columns
plt.figure(figsize=(12, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, len(numerical_columns) // 2 + 1, i)
    sns.boxplot(data=merged_data, y=column)
    plt.title(column)
    plt.ylabel('Values')

plt.tight_layout()
plt.show()


# In[245]:



# Log transformation for 'MONEY SUPPLY' 'UNRATE' 'PRICE' 'JOB GROWTH'

merged_data['MONEY SUPPLY'] = np.log1p(merged_data['MONEY SUPPLY'])
merged_data['UNRATE'] = np.log1p(merged_data['UNRATE'])
merged_data['PRICE'] = np.log1p(merged_data['PRICE'])
merged_data['JOB GROWTH'] = np.log1p(merged_data['JOB GROWTH'])


# # LINEAR REGRESSION MODEL

# In[247]:



# Prepare features and target variable
features = merged_data[[ 'CPI', 'JOB GROWTH', 'MORTGAGE', 'MONEY SUPPLY', 'POPUPLATION', 'UNRATE', 'PRICE']]
target = merged_data['PRICE'] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# # DECISION TREE MODEL

# In[248]:


from sklearn.tree import DecisionTreeRegressor

# Create a Decision Tree model
decision_tree = DecisionTreeRegressor(random_state=42)

# Train the Decision Tree model
decision_tree.fit(X_train, y_train)

# Make predictions
y_pred_dt = decision_tree.predict(X_test)

# Evaluate the Decision Tree model
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Decision Tree - Mean Squared Error: {mse_dt}")
print(f"Decision Tree - R-squared: {r2_dt}")


# # RANDOM FOREST MODEL

# In[249]:


from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest model
random_forest = RandomForestRegressor(random_state=42)

# Train the Random Forest model
random_forest.fit(X_train, y_train)

# Make predictions
y_pred_rf = random_forest.predict(X_test)

# Evaluate the Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - Mean Squared Error: {mse_rf}")
print(f"Random Forest - R-squared: {r2_rf}")


# Linear Regression:
#     
# #Mean Squared Error (MSE): Super close to zero, almost perfect predictions.
# #R-squared: A perfect score, understands the prices perfectly, might be too perfect for new problems.
# 
# Decision Tree:
# 
# #Mean Squared Error (MSE): Small errors, quite accurate but not as perfect as the first model.
# #R-squared: Gets most things right, really good understanding but not flawless.
# 
# Random Forest:
# 
# #Mean Squared Error (MSE): Very tiny errors, incredibly close to perfection.
# #R-squared: Almost perfect, nearly flawless understanding of prices.
# 
# In short, each model predicts house prices. The first is almost too perfect, the second is really good but not perfect, and the third is incredibly close to perfect in guessing prices. They each have strengths, but the last one seems to understand prices the best among them!
# 
# 
# 
# 
# 
# 
# 

# # THANKYOU!
# RITUL PAWAR

# In[ ]:




