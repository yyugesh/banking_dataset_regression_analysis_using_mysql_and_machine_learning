
# Banking Dataset - Regression Analysis using MySQL and Machine Learning

#### This project focuses on predicting key financial metrics from a banking dataset using a regression model. The dataset is stored and managed using a MySQL relational database, and the analysis is performed using Python and machine learning techniques. The goal is to extract insights and build a model that can predict variables such as  Estimated Income, Superannuation Savings, Amount of Credit Cards, Credit Card Balance, Bank Loans, Bank Deposits, Checking Accounts, Saving Accounts, Foreign Currency Account, Business Lending, Properties Owned, Risk Weighting, BRId, GenderId, IAId based on other financial attributes.
## Tech

 **Language**                `Python`

 **Database**                `Mysql`

 **Data Handling**           `pandas`, `numpy`

 **Data Visualization**      `matplotlib`, `seaborn`

 **Machine Learning**        `scikit-learn`

 **Jupyter Environment**     `Jupyter Notebook` or `Google Colab`

 **code editors/IDEs**       `Pycham` or `VS Code (Visual Studio Code)`

 **Version Control**         `git`, `GitHub`
 
 **Environment Management**  `venv`

## Environment Variables

1. What environment variables are needed

2. Why they’re used

### Prerequisites

Make sure you have the following installed:

1. Python 3.7 or higher

2. pip (Python package manager)

3. Git (for cloning the repository)


### Step 1: Clone the Repository

#### Open your terminal or command prompt and run:

git clone https://github.com/yyugesh/banking_dataset_regression_analysis_using_mysql_and_machine_learning.git

### Step 2: Set Up a Virtual Environment

#### Windows
python -m venv venv
venv\Scripts\activate

#### macOS/Linux
python3 -m venv venv
source venv/bin/activate



## Installation & Running the Project

Follow these steps to install and run the Penguin Species Prediction using Machine Learning project on your machine.

### Step 1: Install Required Libraries
    
Run the following commands to install the necessary dependencies:

1. Install **pandas**

    ```
    pip install pandas
    ```

2. Install **numpy**

    ```
    pip install numpy
    ```

3. Install **matplotlib**

    ```
    pip install matplotlib
    ```

4. Install **seaborn**

    ```
    pip install seaborn
    ```

5. Install **scikit-learn**

    ```
    pip install scikit-learn
    ```

Or if you have a requirements.txt file:

    ```
    pip install -r requirements.txt
    ```

### Step 2: Run the project 

Jupyter Notebook, VS Code, or PyCharm, among others are the popular code editors/IDEs.

## Features

- **Banking dataset Regression**: This project utilizes a banking dataset containing a wide range of features related to customer demographics, financial behavior, and banking activity. These features are used to build predictive models (e.g., for income estimation, credit risk, or customer segmentation).

- **Data Preprocessing**: Handles missing values, encodes categorical features, and scales numeric data.

- **Exploratory Data Analysis (EDA)**: Visualizes relationships using pair plots, box plots, and heatmaps.

- **Machine Learning Models**:
  - Linear Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)

- **Model Evaluation**: Assesses performance using metrics like R², MAE, MSE and RMSE Report.

- **Train-Test Split**: Ensures unbiased model evaluation.
## Screenshots

### 1. Pre-processing Data Analysis
![Database Screenshot](Snapshot/Database snapshot.jpg)

### 2. Pre-processing Data Analysis
![PDA Screenshot](Snapshot/sql_banking.png)

### 3. Exploratory Data Analysis
![EDA Screenshot](Snapshot/Figure_2.png)
![EDA Screenshot](Snapshot/Figure_3.png)
![EDA Screenshot](Snapshot/Figure_4.png)
![EDA Screenshot](Snapshot/Figure_5.png)
![EDA Screenshot](Snapshot/Figure_6.png)
![EDA Screenshot](Snapshot/Figure_7.png)
![EDA Screenshot](Snapshot/Figure_8.png)
![EDA Screenshot](Snapshot/Figure_9.png)
![EDA Screenshot](Snapshot/Figure_10.png)
![EDA Screenshot](Snapshot/Figure_11.png)
![EDA Screenshot](Snapshot/Figure_12.png)
![EDA Screenshot](Snapshot/Figure_13.png)
![EDA Screenshot](Snapshot/Figure_14.png)
![EDA Screenshot](Snapshot/Figure_15.png)
![EDA Screenshot](Snapshot/Figure_16.png)
![EDA Screenshot](Snapshot/Figure_17.png)


### 4. Model Prediction Output
![Prediction Result 1](Snapshot/Figure_18.png)

![Prediction Result 2](Snapshot/Figure_19.png)

![Prediction Result 3](Snapshot/Figure_20.png)

![Prediction Result 4](Snapshot/Figure_21.png)

![Prediction Result 5](Snapshot/Figure_22.png)

![Prediction Result 6](Snapshot/Figure_23.png)

![Prediction Result 7](Snapshot/Figure_24.png)

![Prediction Result 8](Snapshot/Figure_25.png)

![Prediction Result 9](Snapshot/Figure_26.png)

![Prediction Result 10](Snapshot/Figure_27.png)

![Prediction Result 11](Snapshot/Figure_28.png)

![Prediction Result 12](Snapshot/Figure_29.png)

![Prediction Result 13](Snapshot/Figure_30.png)

![Prediction Result 14](Snapshot/Figure_34.png)

![Prediction Result 15](Snapshot/Figure_35.png)

![Prediction Result 16](Snapshot/Figure_36.png)

![Prediction Result 17](Snapshot/Figure_37.png)

![Prediction Result 18](SSnapshot/Figure_38.png)


### 5. Model Evaluation Comparison Output
![Model Accuracy](Snapshot/Model Evaluation.png)

## Demo

Check out the demo of the Penguin Species Prediction app:

[Click here to try the app](https://github.com/yyugesh/banking_dataset_regression_analysis_using_mysql_and_machine_learning/blob/6f0912a3309dc755390b1438ff17e457d0f35b80/Banking%20Dataset%20-%20Regression%20Analysis%20using%20MySQL%20and%20Machine%20Learning.gif)  

## About the Dataset

- Dataset Source: The dataset used in this project is stored locally and named Banking.csv. It contains anonymized banking customer data for educational and analytical purposes.

- Target Variable: The main target for prediction in this project is Estimated Income, which is modeled using various customer features.

- Data Privacy: Sensitive customer identifiers such as names and IDs are included only for structural completeness and are not used in modeling.

## Assumptions:

- All numeric fields are assumed to be clean and properly scaled during preprocessing.

- Categorical features are label-encoded or one-hot encoded based on the model used.

- Limitations: This project is designed for academic and demonstration purposes only and may not reflect real-world production constraints or compliance standards.

- Related Tools: Data was accessed, analyzed, and modeled using tools like MySQL, pandas, scikit-learn, and matplotlib in a Python environment.