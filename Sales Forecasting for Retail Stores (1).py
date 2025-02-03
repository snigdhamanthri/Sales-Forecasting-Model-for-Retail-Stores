#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the data
def load_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("Data Loading Complete:")
    print("Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    
    return df

# Example usage
df = load_data("C:\\Users\\Snigdha\\OneDrive\\Desktop\\retail_sales_dataset.csv")


# In[95]:


def clean_and_preprocess(df):
    """
    Clean the data and perform feature engineering
    """
    # Make a copy of the dataframe
    df_processed = df.copy()
    
    # Check for missing values
    print("\nMissing Values:")
    print(df_processed.isnull().sum())
    
    # Check for duplicates
    duplicates = df_processed['Transaction ID'].duplicated().sum()
    print(f"\nDuplicate Transactions: {duplicates}")
    
    # Handle outliers using IQR method
    def handle_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        print(f"Outliers in {column}: {outliers}")
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Check outliers in numerical columns
    numerical_cols = ['Age', 'Quantity', 'Price per Unit', 'Total Amount']
    for col in numerical_cols:
        df_processed = handle_outliers(df_processed, col)
    
    # Feature Engineering
    # Extract date components
    df_processed['Year'] = df_processed['Date'].dt.year
    df_processed['Month'] = df_processed['Date'].dt.month
    df_processed['Day'] = df_processed['Date'].dt.day
    df_processed['DayOfWeek'] = df_processed['Date'].dt.dayofweek
    
    # Create seasonal indicators
    df_processed['Season'] = df_processed['Month'].apply(lambda x: 
        'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else 'Fall'
    )
    
    # Create age groups
    df_processed['AgeGroup'] = pd.cut(
        df_processed['Age'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=['18-24', '25-34', '35-44', '45-54', '55+']
    )
    
    print("\nShape after preprocessing:", df_processed.shape)
    return df_processed

# Process the data
df_processed = clean_and_preprocess(df)


# In[96]:


def perform_eda(df):
    """
    Perform Exploratory Data Analysis
    """
    # 1. Sales Trends
    plt.figure(figsize=(15, 10))
    
    # Monthly sales trend
    plt.subplot(2, 2, 1)
    monthly_sales = df.groupby(df['Date'].dt.strftime('%Y-%m'))['Total Amount'].sum()
    plt.plot(range(len(monthly_sales)), monthly_sales.values, marker='o')
    plt.title('Monthly Sales Trend')
    plt.xticks(range(len(monthly_sales)), monthly_sales.index, rotation=45)
    plt.grid(True)
    
    # Sales by category
    plt.subplot(2, 2, 2)
    category_sales = df.groupby('Product Category')['Total Amount'].sum()
    category_sales.plot(kind='bar')
    plt.title('Sales by Category')
    plt.xticks(rotation=45)
    
    # Sales by season
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='Season', y='Total Amount')
    plt.title('Sales Distribution by Season')
    plt.xticks(rotation=45)
    
    # Customer age distribution
    plt.subplot(2, 2, 4)
    sns.histplot(data=df, x='Age', bins=30)
    plt.title('Customer Age Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation Analysis
    numerical_cols = ['Age', 'Quantity', 'Price per Unit', 'Total Amount']
    correlation = df[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Perform EDA
perform_eda(df_processed)


# In[97]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

def prepare_data_for_modeling(df):
    """
    Prepare data for modeling
    """
    # Prepare features for Random Forest
    features = ['Month', 'Day', 'DayOfWeek']
    X = df[features]
    y = df['Total Amount']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def build_models(df):
    """
    Build and evaluate models
    """
    # Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data_for_modeling(df)
    
    # 1. Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Calculate metrics for Random Forest
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mape = mean_absolute_percentage_error(y_test, rf_pred)
    
    print("Random Forest Metrics:")
    print(f"RMSE: {rf_rmse:.2f}")
    print(f"MAPE: {rf_mape:.2%}")
    
    # 2. SARIMA Model
    # Prepare time series data
    monthly_sales = df.groupby('Date')['Total Amount'].sum().resample('M').sum()
    
    # Fit SARIMA model
    sarima_model = SARIMAX(monthly_sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_results = sarima_model.fit()
    
    # Make predictions
    forecast = sarima_results.get_forecast(steps=6)
    forecast_mean = forecast.predicted_mean
    
    print("\nSARIMA Model Summary:")
    print(sarima_results.summary())
    
    return rf_model, sarima_results, forecast_mean

# Build and evaluate models
rf_model, sarima_model, forecast = build_models(df_processed)


# In[98]:


def generate_insights(df, rf_model, forecast):
    """
    Generate insights and recommendations
    """
    # Sales patterns
    monthly_sales = df.groupby(df['Date'].dt.strftime('%Y-%m'))['Total Amount'].sum()
    
    # Plot actual vs predicted sales
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales.index, monthly_sales.values, label='Actual Sales')
    plt.plot(range(len(monthly_sales), len(monthly_sales) + len(forecast)), 
             forecast, 'r--', label='Forecast')
    plt.title('Sales Forecast')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
    
    # Generate insights
    print("\nKey Insights:")
    print("1. Sales Patterns:")
    print(f"   - Peak Sales Month: {monthly_sales.idxmax()}")
    print(f"   - Lowest Sales Month: {monthly_sales.idxmin()}")
    
    print("\n2. Product Categories:")
    category_sales = df.groupby('Product Category')['Total Amount'].sum()
    print(f"   - Best Performing Category: {category_sales.idxmax()}")
    print(f"   - Category Sales Distribution:")
    for category, sales in category_sales.items():
        print(f"     {category}: ${sales:,.2f}")
    
    print("\n3. Customer Demographics:")
    age_group_sales = df.groupby('AgeGroup')['Total Amount'].mean()
    print(f"   - Highest Spending Age Group: {age_group_sales.idxmax()}")
    
    print("\n4. Seasonal Trends:")
    season_sales = df.groupby('Season')['Total Amount'].mean()
    print(f"   - Best Performing Season: {season_sales.idxmax()}")
    
    print("\n5. Recommendations:")
    print("   - Inventory Planning: Adjust stock levels based on seasonal patterns")
    print("   - Marketing Strategy: Target high-value customer segments")
    print("   - Promotion Planning: Align promotions with predicted sales peaks")

# Generate insights and recommendations
generate_insights(df_processed, rf_model, forecast)


# In[ ]:




