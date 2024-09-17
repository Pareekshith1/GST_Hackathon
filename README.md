# GST-Predictor (Version -1.0)
#### Project for GST-Hackathon

Welcome to the GST-Predictor project! This repository contains code and resources for a predictive modeling project aimed at identifying key features from medical data. This README file will guide you through the project, providing detailed explanations of its components, methodologies, and results.

## Project Overview

The GST-Predictor project aims to build a robust machine learning model to predict target variables from the given data using various algorithms and techniques. This version of the project focuses on evaluating and deploying a CatBoostClassifier model for binary classification tasks.

## Features

- **Data Loading**: Load and preprocess data from CSV files.
- **Model Training**: Train a CatBoostClassifier using randomized search for hyperparameter optimization.
- **Model Evaluation**: Assess model performance using metrics such as accuracy, precision, recall, F1 score, log loss, balanced accuracy, and AUC-ROC.
- **Visualizations**: Generate and visualize ROC curves, Precision-Recall curves, and feature importance plots.

## Data

The project uses GST data in CSV format. The datasets include:

- **Training Data**: Contains features and target labels for model training.
- **Testing Data**: Used to evaluate the model’s performance.

### Data Files

- `X_Train_Data_Input.csv`: Features for training.
- `Y_Train_Data_Target.csv`: Target labels for training.
- `X_Test_Data_Input.csv`: Features for testing.
- `Y_Test_Data_Target.csv`: Target labels for testing.

## Installation

To run this project, you need to set up your environment and install the required dependencies. Here’s how:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/gst-predictor.git
   cd gst-predictor
   
2. **Navigate to the the project folder**

    ```bash
   cd GST-Predictor-0.1

3. **Install Dependencies**
    
    ```bash
   pip install -r requirements.txt

## Model 
### CatBoost Algorithm
CatBoost, short for Categorical Boosting, is a gradient boosting algorithm 
developed by Yandex. It is specifically designed to handle categorical 
features and is known for its efficiency, high accuracy, and ability to 
prevent overfitting. CatBoost utilizes a unique Ordered Boosting approach, 
which ensures that the model remains robust by preventing the overfitting of 
training data. This algorithm is particularly suitable for tasks involving large 
datasets and complex patterns.

### Why CatBoost?
**Handles Missing Values:** CatBoost naturally handles missing values, eliminating the need for extensive preprocessing or imputation.

**Prevents Overfitting:** The Ordered Boosting technique used by CatBoost reduces the risk of overfitting, ensuring the model generalizes well to unseen data.

**Optimized for Categorical Data:** CatBoost can directly process categorical features, automatically converting them into numerical representations, which preserves the information and relationships within the data.

**High Accuracy and Efficiency:** CatBoost is both accurate and fast, making it suitable for large-scale datasets, providing reliable results without excessive computational resources.

**Balances Overfitting and Underfitting:** By using regularization techniques, CatBoost finds the right balance, making it a robust choice for various predictive tasks.

## The Training Models

 1) We have implemented 2 optimized CatBoost training model which adapts/ adjust itself according 
to the given dataset 
2) These models will analyze the dataset and sets the parameter accordingly so that the model can process the dataset more clearly and accurately.
3) They auto adjust the parameter by itereating over the different predefined best fit parameter and selects the one which is suit the most for processing 
4) We have implimented 2 types of search algorithm and assigned it to the model .

### Randomized Search Training Model Using Catboost Algorithm
Randomized Search is a hyperparameter optimization technique used to find the best set of hyperparameters for a machine learning model
Randomized Search samples a fixed number of hyperparameter combinations from a specified distribution.

##### How It Benefits the Model
1. #### Efficiency:
Speed: Can be faster than exhaustive search methods because it only evaluates a subset of possible hyperparameter combinations.
Resource Management: More computationally efficient, making it feasible to run on limited resources or within a constrained time frame.
Exploration:

2. #### Diverse Searches:
Random sampling allows for exploring a wider range of hyperparameter values, which can potentially uncover better combinations.
Flexibility: Handles various types of hyperparameters (continuous, categorical) and distributions (uniform, log-uniform).

3. #### Scalability:
Large Parameter Spaces: Suitable for scenarios with a large number of hyperparameters or wide ranges of values, where exhaustive search would be impractical.

4. #### Reduced Overfitting Risk:
Less Overfitting: By not exhaustively searching all possible combinations, it reduces the risk of overfitting the validation data.

### Grid Search Training Model Using Catboost Algorithm
Grid Search is a systematic approach for hyperparameter 
tuning in machine learning models. It involves specifying 
a grid of hyperparameter values to explore and evaluates 
the model’s performance for every possible combination 
within this grid. By doing so, Grid Search aims to find 
the optimal set of hyperparameters that yields the best 
performance according to a predefined metric, such as accuracy or 
F1 score. While it provides a thorough search and guarantees finding
the best configuration within the grid, it can be computationally
expensive and time-consuming, especially when dealing with a large 
number of hyperparameters or extensive ranges.

##### How It Benefits the Model

1. #### Comprehensive Search:
Thorough Evaluation: Evaluates all possible combinations of hyperparameters, ensuring that the best possible combination within the defined grid is found.
Guaranteed Optimal Solution: If the grid is sufficiently fine and covers the parameter space well, it guarantees finding the best combination within the specified range.

2. #### Simplicity:
Easy to Implement: Straightforward to set up and understand, making it accessible for practitioners without needing sophisticated optimization techniques.

3. #### Consistency:
Stable Results: Provides consistent results as the search is deterministic, meaning the same grid will yield the same results every time it’s run.

4. #### Visualization:
Clear Results: Allows for easy visualization of how different hyperparameters affect model performance by evaluating all combinations.

## Usage
### Training the Model

Before training the model select the algorithm which you want from the 2 given algorithm which are Random-Search Algorithm and Grid Search Algorithm 
### The Grid Search algorithm 
Iterates through every possible combinations of the parmater and finds the best parameter suitable for the model to process the data accurately.
This model is a little bit Exhaustive as th processing time is high but iterates through each and every combination to find out the best parameter for the model.

To run the code 
```bash
 python Exhaustive_Grid_Search.py
```

### The Randomized Search algorithm
The randomized search algorithm takes random combination and finds the best parameters for the model from those combinations.
This model ont be going through each and every combination excepts it makes its own random combinations and finds the best parametes from it 
THis model is more faster but wont iterate through every combinations.

To run the code 
```bash
 python Randomized_Search_Model.py
```

### Testing Model 
The testing model leverages the best parameters identified during the 
hyperparameter tuning phase to evaluate the final CatBoostClassifier on 
a separate test dataset. It predicts class labels and computes probabilities,
which are then used to generate performance metrics such as accuracy, 
precision, recall, F1 score, and AUC-ROC. These metrics provide insights 
into the model's effectiveness and generalization ability on unseen data.

```bash
    python GST-Predictor (Version -1.0).py
```

## Metrics
### Metrics Used
**Accuracy**: The proportion of correct predictions.

**Precision**: The proportion of positive identifications that were actually correct.

**Recall**: The proportion of actual positives that were correctly identified.

**F1 Score**: The harmonic mean of precision and recall.  

**Log Loss**: Measures the performance of a classification model where the outcome is a probability value.  

**Balanced Accuracy**: The average of recall obtained on each class.  

**AUC-ROC**: Area under the Receiver Operating Characteristic curve, indicating the model’s ability to discriminate between classes.

### Performance Results
**Accuracy** : 0.9784

**Precision**: 0.8485

**Recall**: 0.9379

**F1 Score**: 0.8910

**Log Loss**: 0.0491

**Balanced Accuracy**: 0.9602

**AUC-ROC**: 0.9949

![AUC-ROC Curve](./accuracy.png?raw=true "Auc-Roc")

## Visualizations
### ROC Curve

The ROC curve visualizes the model’s performance. The AUC-ROC score is 0.9949, indicating excellent model performance.

![AUC-ROC Curve](./aucroccurve.png?raw=true "Auc-Roc")


### Precision-Recall Curve

The Precision-Recall curve shows the trade-off between precision and recall for different threshold values.

![AUC-ROC Curve](./precision.png?raw=true "Precision")

### Feature Importance

The Feature Importance plot displays the significance of each feature used in the model.

![AUC-ROC Curve](./featureimportance.png?raw=true "Featureimportance")

## Citations
#### 1) CatBoost Documentation
#### 2) Plotly Documentation

## Plagiarism Declaration
I, hereby declare that this project, including all code, design, and documentation, is my original work. No part of this project has been copied or plagiarized from any other source without proper attribution.

Any external libraries, tools, or references that were utilized during the development of this project have been appropriately cited. In particular, the use of CatBoost and Plotly libraries is acknowledged, and their respective documentations have been consulted to enhance the project:

**CatBoost Documentation:** Used to understand and implement the CatBoost model and its features.

**Plotly Documentation:** Utilized for generating visualizations, such as the AUC-ROC curve and other performance metrics.

I affirm that this work has been created with academic integrity and adheres to ethical guidelines, ensuring all contributions are properly credited.
