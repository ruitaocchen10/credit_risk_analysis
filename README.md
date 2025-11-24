# Credit Risk Analysis

Predicting loan defaults using the Home Credit Default Risk dataset from Kaggle.

# Tech Stack

Databricks - Scalability for large data set
Python - Main language
pandas - Data manipulation and analysis
PySpark - Big data processing
scikit-learn - Logistic regression, preprocessing (StandardScaler), train/test split, metrics
matplotlib & seaborn - Visualizations

## The Data Pipeline

1. **Data Cleaning** (`data_wrangling.ipynb`)
   - Dropped 48 columns with >40% missing values (mostly building features)
   - Imputed missing values using median/mode/zeros depending on the feature
   - Cleaned up rare categorical values
   - One-hot encoded all categorical variables

2. **Data Normalization** (`data_normalization.ipynb`)
   - Fixed the DAYS_EMPLOYED anomaly (365243 → 0 for unemployed)
   - Created IS_UNEMPLOYED flag
   - Applied StandardScaler to numerical features

3. **Logistic Regression** (`logistic_regression.ipynb`)
   - 80/20 train/test split
   - Trained with balanced class weights
   - Evaluated with accuracy, precision, recall, F1, ROC-AUC

## Key Decisions

- **Missing values**: Dropped high-missing columns (>40%), imputed the rest
- **Unemployed handling**: Converted that weird 365243 value to 0 with a separate flag
- **Encoding**: One-hot encoding for all categoricals (most had <10 categories anyway)
- **Scaling**: StandardScaler for logistic regression

## Dataset

~307K loan applications, ~100 features after cleaning. Target is binary (0=no default, 1=default).

Original data: [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data?select=application_train.csv)

The logistic regression model achieved decent performance on the test set:

ROC-AUC: ~0.74 - Pretty good at distinguishing defaults from non-defaults
Accuracy: High (~92%), but this is misleading due to class imbalance (only ~8% default rate)

Top predictive features: EXT_SOURCE_2 and EXT_SOURCE_3 (external credit scores) were by far the most important, along with days employed, age, and income-related features.
Main takeaway: The model does a reasonable job, but there's room for improvement with more advanced models and better handling of the class imbalance.
