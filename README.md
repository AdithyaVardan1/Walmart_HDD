# OG_Walmart

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Code Structure](#code-structure)
  - [Data Preparation](#data-preparation)
  - [Model Training and Model Evaluation](#model-training-and-model-evaluation)
- [References](#references)

## Introduction
OG_Walmart is a predictive modeling project that leverages machine learning to analyze and predict hard drive failures based on historical data collected from hard drives monitored Backblaze. Utilizing data from 2019-01-01 to 2019-04-30, this project employs Random Forest Classifier among other techniques to forecast potential failures, aiming to enhance maintenance schedules and reduce downtime.

## Installation
To set up the project locally, follow these steps:

- git clone https://github.com/AdithyaVardan1/Walmart_Random_forest.git
- cd Walmart_Random_forest
- pip install -r requirements.txt

## Code Structure
This section outlines the main components of the codebase and their functionality.

### Data Preparation
- **final_cleaning.ipynb**: Contains code for cleaning and preprocessing the dataset. It includes functions to handle missing values, normalize data, and split the dataset into training and testing sets.

### Model Training and Model Evaluation
- **classifier.ipynb**: Implements the RandomForestClassifier model. It includes code for training the model on the preprocessed data, tuning model parameters using RandomizedSearchCV, and saving the trained model for later use.And Contains code to evaluate the trained model's performance on the test set. It outputs metrics such as accuracy, precision, recall, and F1 score to assess the model's effectiveness in predicting hard drive failures.

## References
This project was inspired by and built upon the concepts, datasets, and code from the following sources:

- **Scientific Paper**: ["Interpretable predictive maintenance for hard drives"](https://www.sciencedirect.com/science/article/pii/S2666827021000219) - This paper discusses the methodologies, algorithms, and impacts of predictive maintenance on hard drive failures, providing a foundational theory for our project's approach.

Please note that while we have drawn upon these resources for inspiration and guidance, the implementations and any modifications to the algorithms or code are our own, tailored to the specific requirements and objectives of our project.



