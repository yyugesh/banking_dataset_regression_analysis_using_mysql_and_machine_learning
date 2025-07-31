import mysql.connector as sql_conn
# import getpass
# # Ask for the password securely
# Password = getpass.getpass("Enter Your Mysq Password: ")
# print(Password)

# Create the connection Object
cnx = sql_conn.connect(host = "localhost",
                       username = "root",
                       password = "")

print("My Sql executed successfully")

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Query to open a "Banking" dataset from the Mysql DataBase
query = "select* from banking_db.Customer"

# Data Collection:

mysql_df = pd.read_sql(query, cnx)             # Read the dataset from DB or load from MySQL
cnx.close()                                    # Close the MySql connection


# Data Understanding
print(mysql_df.head())
print("-"* 42)
print(mysql_df.shape, mysql_df.info(), mysql_df.describe())

# Drop Irrelevant Columns
df = mysql_df.drop(columns=['ï»¿Client ID', 'Name', 'Joined Bank', 'Banking Contact'])

# Separate Target and Features
X = df.drop('Bank Deposits', axis=1)  # All except target / Independent variables [features / inputs]
y = df['Bank Deposits']                     # Dependent variable [Target / Output / Label]

# Identify Categorical and Numerical Columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# OneHotEncoding and Scaling
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('scale', StandardScaler(), numerical_cols)
])

# Define Regression Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "KNN Regressor": KNeighborsRegressor()
}

'''
 Created a models dictionary with key name ML models strings, and 
 values are initialized through model objects from scikit-learn.
'''

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train            [Feature data used to train the model]
# X_test             [Feature data used to test the model (after training)]
# y_train            [Target values for the training dataset]
# y_test             [Target values for the test dataset]
# X                  [Features (independent variables)]
# y                  [Target variable (dependent variable)]
# test_size=0.2	     [20% of the data goes to the test, and 80% go for training set]
# random_state=42	 [Controls the random split. Using the same number gives you reproducible results every time.]

# Evaluation Function
def evaluate_model(y_true, y_pred):
    return {
        "R²": round(r2_score(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "MSE": round(mean_squared_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
    }

# Train and Evaluate All Models
results = {}                                       # Dictionary to store R² scores
predictions = {}                                   # Dictionary to store predictions
for name, model in models.items():                 # Loop through each model
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)                           # Train the model
    y_pred = pipeline.predict(X_test)                        # Predict on test data
    results[name] = evaluate_model(y_test, y_pred)
    predictions[name] = y_pred

# Show Evaluation Results
results_df = pd.DataFrame(results).T
'''Converts nested dictionary to DataFrame, 
   Transposes rows ↔ columns'''
print("\nModel Evaluation Summary:\n")
print(results_df)

# UNIVARIATE ANALYSIS (One Column at a Time)

plt.figure(figsize=(10, 6))
sns.histplot(df['Bank Deposits'], bins=30, kde=True)
plt.title("Distribution of Bank Deposits")
plt.xlabel("Deposit Amount")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Numerical Features
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# BIVARIATE ANALYSIS (Target vs Features)

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=df[col], y=df['Bank Deposits'])
    plt.title(f"{col} vs Bank Deposits")
    plt.xlabel(col)
    plt.ylabel("Bank Deposits")
    plt.grid(True)
    plt.show()

# Actual vs Predicted for All Models

for name, y_pred in predictions.items():
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{name}: Actual vs Predicted")
    plt.xlabel("Actual Bank Deposits")
    plt.ylabel("Predicted Bank Deposits")
    plt.grid(True)
    plt.show()

# print("R\u00b2 :")