#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

# Create a dictionary with dummy data
data = {
    'ID': range(1, 101),
    'Name': ['Person_' + str(i) for i in range(1, 101)],
    'Age': np.random.randint(18, 60, size=100),
    'Salary': np.random.randint(30000, 120000, size=100),
    'Department': np.random.choice(['HR', 'Finance', 'IT', 'Marketing'], size=100)
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('/Users/madhavibhat/Documents/Learning Python/Python-py-Files/dummy_data.csv', index=False)


# In[5]:


print (data)


# In[6]:


sample_df = df.sample(n=10)


# In[7]:


print(sample_df)


# In[8]:


# Display the first few rows of the dataframe
print(df.head())


# In[9]:


print(df.describe)


# In[10]:


print(df.describe())


# In[11]:


print(df.isnull().sum())


# In[19]:


# Group by Department and calculate the mean salary
mean_salary_by_dept = df.groupby('Department')['Salary'].mean()
print(mean_salary_by_dept)


# In[21]:


mean_salary_by_dept.plot(kind='bar', color='skyblue')
plot.show()


# In[23]:


import matplotlib.pyplot as plt


# In[24]:


mean_salary_by_dept.plot(kind='bar', color='skyblue')
plt.show()


# In[25]:


# Plotting the mean salary by department
plt.figure(figsize=(10, 6))
mean_salary_by_dept.plot(kind='bar', color='skyblue')
plt.title('Mean Salary by Department')
plt.xlabel('Department')
plt.ylabel('Mean Salary')
plt.show()


# In[30]:


# Group by Department,Age and calculate the mean salary
mean_salary_by_dept_age = df.groupby(['Department','Age'])['Salary'].mean()
print(mean_salary_by_dept_age)



# In[33]:


# Plotting the mean salary by department,Age
plt.figure(figsize=(10, 6))
mean_salary_by_dept_age.plot(kind='bar', color='skyblue')
plt.title('Mean Salary by Department and age')
plt.xlabel('Department')
plt.ylabel('Mean Salary')
plt.show()


# In[34]:


import seaborn as sns


# In[35]:


# Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[36]:


# Scatter plot of Age vs Salary
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Salary', data=df, hue='Department')
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend(title='Department')
plt.show()


# In[37]:


# Correlation matrix
corr_matrix = df.corr()
print(corr_matrix)


# In[38]:


# Correlation matrix
# Select only numeric columns for correlation, correalation matrix does not work on non-numeric data like persons
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
print(corr_matrix)


# In[39]:


# Heatmap of the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[53]:


# Display the first few rows of the dataframe
print(df.head())

# Print the columns of the dataframe to verify
print(df.columns)


# In[54]:


df = pd.DataFrame(data)


# In[55]:


# Display the first few rows of the dataframe
print(df.head())

# Print the columns of the dataframe to verify
print(df.columns)


# In[56]:


# Check if 'Department' column is in the DataFrame
if 'Department' in df.columns:
    # One-hot encode the categorical variable 'Department'
    df = pd.get_dummies(df, columns=['Department'], drop_first=True)
else:
    print("The 'Department' column is not found in the DataFrame.")


# In[63]:


# Define features and target variable, Department_Finance is excluded because it doen't have any values 
X = df[['Age', 'Department_HR', 'Department_IT', 'Department_Marketing']]
y = df['Salary']


# In[67]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[68]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[69]:


# Create a linear regression model
regressor = LinearRegression()


# In[70]:


# Train the model
regressor.fit(X_train, y_train)


# In[71]:


# Predict on the test set
y_pred = regressor.predict(X_test)


# In[72]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[73]:


print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[74]:


# Display the regression coefficients
coefficients = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coefficients)


# In[77]:


# Function to predict salary
def predict_salary(age, department):
    # Create a dictionary for the input data
    input_data = {
        'Age': [age],
        'Department_HR': [1 if department == 'HR' else 0],
        'Department_IT': [1 if department == 'IT' else 0],
        'Department_Marketing': [1 if department == 'Marketing' else 0]
    }
    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame(input_data)
    # Predict the salary
    predicted_salary = regressor.predict(input_df)
    return predicted_salary[0]


# In[78]:


# Example usage
age = 35
department = 'IT'
predicted_salary = predict_salary(age, department)
print(f"The predicted salary for a {age}-year-old in the {department} department is ${predicted_salary:.2f}")


# In[ ]:




