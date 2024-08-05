
# Singapore_Resale_Flat_Price_Prediction

# Problem Statement:
The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.


# Project Overview
This project aims to analyze and predict the resale prices of HDB flats in Singapore using data analysis and machine learning techniques.

# Data Source
The dataset is sourced from the Singapore government's open data portal, specifically from the Housing and Development Board (HDB). It includes information on resale transactions such as:
Month:The month of the resale transaction.
Town: The town where the flat is located.
Flat Type: The type of flat (e.g., 3-room, 4-room, Executive).
Flat Model: The model of the flat.
Floor Area (sqm): The floor area of the flat in square meters.
Lease Commence Date: The year the lease commenced.
Remaining Lease: The remaining lease in years.
Resale Price: The resale price of the flat.

# Data Preprocessing
Handling Missing Values: Identify and handle missing data.
Feature Engineering: Create new features like the age of the flat and floor area per room.
Encoding Categorical Variables: Convert categorical variables into numerical values using one-hot encoding.
Data Normalization: Scale numerical features for improved model performance.

# Model Training and Evaluation
Data Splitting: Split the dataset into training and testing sets.
Model Training: Train models using cross-validation.
Evaluation Metrics: Use Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared to evaluate models.
