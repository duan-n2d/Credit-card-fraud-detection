# Credit-card-fraud-detection
This repository contains code for detecting credit card fraud using various machine learning models. The dataset used in this project is sourced from the [Credit Card Fraud](https://www.kaggle.com/datasets/whenamancodes/fraud-detection).

## Table of Contents
[Introduction](introduction)
[Data Preprocessing](data-preprocessing)
[Feature Selection](feature-selection)
[Model Training and Evaluation](model-training-and-evaluation)
[Results](results)
[Installation](installation)
[Usage](usage)
[Contributing](contributing)
[License](contributing)

## Introduction
Credit card fraud detection is a crucial task in financial security. This project aims to identify fraudulent credit card transactions using various machine learning algorithms, including Logistic Regression, Isolation Forest, Random Forest, CatBoost, Deep Neural Networks (DNNs), XGBoost, and Long Short-Term Memory (LSTM) networks.

## Data Preprocessing
Data preprocessing steps include:
* Exploratory Data Analysis (EDA): Understanding the dataset by examining its structure, summary statistics, and distribution of the target variable.
* Correlation Analysis: Identifying the relationship between features and the target variable. Features with a correlation threshold of 0.1 are selected for further analysis.
* Data Cleaning: Removing duplicate rows and handling missing values.
* Feature Scaling: Standardizing features using StandardScaler to bring all features to the same scale.
* Train-Test Split: Splitting the dataset into training (80%) and testing (20%) sets.

## Feature Selection
Features with a correlation greater than 0.1 with the target variable are selected. This helps in reducing dimensionality and improving the model's performance. The following features are selected:
V1, V3, V4, V7, V10, V11, V12, V14, V16, V17, V18

## Model Training and Evaluation
I train and evaluate several models:
* Logistic Regression
* Isolation Forest
* Random Forest
* CatBoost
* Deep Neural Networks (DNNs)
* XGBoost
* Long Short-Term Memory (LSTM) Networks
=> Each model is trained on the training set and evaluated on the test set. Performance metrics include Accuracy, Precision, Recall, F1 Score, and AUC-ROC.

## Results
The results of each model are summarized and compared based on their performance metrics. ROC curves are plotted to visualize the performance of each model.

## Installation
To run this project, you need to have Python installed along with the following libraries:
```
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
catboost
keras
tensorflow
```
You can install the required libraries using:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost catboost keras tensorflow
```

## Usage
Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
```
Navigate to the project directory:
```bash
cd credit-card-fraud-detection
```
Run the main script to preprocess the data, train the models, and evaluate their performance:
```bash
python main.py
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](#license) file for details.
