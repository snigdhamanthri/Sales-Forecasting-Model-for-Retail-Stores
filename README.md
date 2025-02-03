# Sales-Forecasting-Model-for-Retail-Stores
This repository contains a comprehensive sales forecasting model for retail stores using historical sales data. The model combines Random Forest and SARIMA approaches to predict future sales trends and provide actionable business insights.
## Overview
This repository contains a sales forecasting model designed to enhance revenue predictions for retail stores. The project employs Random Forest Regression and SARIMA models to predict future sales, enabling better decision-making in inventory planning, marketing strategies, and promotional activities.
## Problem Statement
The goal is to develop a machine learning model that can accurately forecast monthly sales for retail stores by considering:

Historical sales patterns
Seasonal trends
Customer demographics
Product category performance
## Data Preprocessing
Data Cleaning:

Checked and handled missing values
Removed duplicate transactions
Handled outliers using IQR method
Feature engineering for temporal components


Feature Engineering:

Extracted date components (Year, Month, Day)
Created seasonal indicators
Generated age groups
Added day-of-week features
## Models
Random Forest Regressor:

Features: Month, Day, DayOfWeek
Train/Test Split: 80/20
Scaled features using StandardScaler


SARIMA Model:

Parameters: (1,1,1) with seasonal order (1,1,1,12)
Monthly aggregated data
6-month forecast horizon
## Key Features
The analysis revealed important patterns in:

Temporal Patterns:

Monthly sales trends
Day-of-week effects
Seasonal variations


Product Categories:

Clothing: 351 transactions
Electronics: 342 transactions
Beauty: 307 transactions


Customer Demographics:

Gender distribution (Female: 51%, Male: 49%)
Age group analysis
Purchase behavior patterns
## Model Performance
The models achieved the following metrics:

Random Forest:

RMSE and MAPE scores provided


SARIMA:

Detailed forecast with confidence intervals
Model summary statistics
## Key Insights and Recommendations
Sales Patterns:

Identified peak and lowest sales months
Category-wise performance analysis
Seasonal trend impacts


Business Recommendations:

Inventory Planning: Based on seasonal patterns
Marketing Strategy: Target high-value segments
Promotion Planning: Align with predicted peaks
## Files in Repository
retail_sales_dataset.csv: Source data
Sales Forecasting for Retail Stores.py: Main implementation
Generated visualizations:

retail_sales_dashboard.png
retail_detailed_insights.png
retail_enhanced_insights.png
## Usage
Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

Run the analysis:

python "Sales Forecasting for Retail Stores.py"

## Visualizations
The code generates multiple visualization dashboards:

Monthly sales trends
Category performance
Customer demographics
Seasonal patterns
Correlation analysis
## Future Enhancements
Implement more advanced forecasting models
Add external factors (economic indicators, weather)
Create interactive dashboards
Incorporate real-time data updates

This project provides a foundation for data-driven decision-making in retail sales forecasting and can be extended based on specific business needs.
