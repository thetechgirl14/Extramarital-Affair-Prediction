# Logistic Regression for Predicting Extramarital Affairs

This project aims to predict the likelihood of relationship affairs using logistic regression. It includes data preprocessing, data visualization, model training, and evaluation.

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)

## Project Description

The project utilizes the logistic regression algorithm to predict relationship affairs based on various features such as marriage rating, age, years married, children, religiousness, education, and occupation. The dataset used is the "Fair's Extramarital Affairs" dataset from the Statsmodels library.

The project consists of the following components:
- Data Preprocessing: The dataset is preprocessed to handle missing values and create binary labels for affairs.
- Data Visualization: Visualizations are created to explore the relationship between different features and the occurrence of affairs.
- Model Training: Logistic regression model is trained using the preprocessed dataset.
- Model Evaluation: The trained model is evaluated using accuracy, confusion matrix, classification report, and ROC curve.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your_username/relationship-affairs-prediction.git
   ```

   Change Directories to Project Directory
   ```
   cd relationship-affairs-prediction
   ```

3. Create a virtual environment (optional):

```
python -m venv env
```
For Linux/Mac
```
source env/bin/activate 
```
For Windows
```
env\Scripts\activate 
```
3. Install the dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Run the main.py file to execute the project:
```
python main.py
```

2. The program will perform data preprocessing, data visualization, model training, and evaluation. The results will be displayed in the console and relevant figures will be saved in the `figures` directory.

## Requirements

The project requires the following dependencies:
- numpy==1.21.0
- pandas==1.3.0
- statsmodels==0.12.2
- matplotlib==3.4.3
- seaborn==0.11.1
- scikit-learn==0.24.2
- imbalanced-learn==0.8.0

You can install all the dependencies by running the command:
```
pip install -r requirements.txt
```
   

