# Heart-Disease-Detection

Heart disease detection using hybrid machine learning / ensemble learning involves developing models to predict the presence or risk of heart disease based on various clinical and demographic data. The goal is to improve early diagnosis and facilitate timely treatment, potentially reducing the morbidity and mortality associated with heart disease.

# Research Done
If you are intrested in Research Aspect of this projevct then you can read my Research at - http://dx.doi.org/10.13140/RG.2.2.15033.38247

# Heart Disease Prediction

This project involves building and evaluating multiple base estimators as machine learning models to predict the presence of heart disease based on various medical attributes. The dataset used is `heart.csv`.
The project is fully deployed on render so anyone could access the online-heart-disease-predictor ensure the website loads properly as it takes sometime to load if detected any traffic.To visit follow the link
 https://heart-disease-monitoring.onrender.com/

## Table of Contents

- [Overview](#overview)
- [Dataset Information](#dataset-information)
- [Installation](#installation)
- [Usage](#usage)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Visualization](#visualization)

- [Contributing](#contributing)
- [License](#license)

  <p align="center">
  <img src="https://github.com/erenyeager101/ML-based-Heart-Disease-Detection/blob/main/pics/Q6_Image1_EDAI2.png" alt="Heart Disease Detection Webpage" width="600" />
</p>


## Video-Overview

The goal of this project is to predict the presence of heart disease in a patient based on several medical features. We evaluate multiple machine learning algorithms to determine the most accurate model.

## Overview
[![Machine Learning based Heart Disease detection model](https://img.youtube.com/vi/3Txp134133s/0.jpg)](https://www.youtube.com/watch?v=3Txp134133s)

If you want to view the project deatis on youtube watch it here -> https://youtu.be/3Txp134133s

## Dataset Information

The dataset contains the following columns:

1. `age`: Age of the patient
2. `sex`: 1 = male, 0 = female
3. `cp`: Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)
4. `trestbps`: Resting blood pressure
5. `chol`: Serum cholesterol in mg/dl
6. `fbs`: Fasting blood sugar > 120 mg/dl
7. `restecg`: Resting electrocardiographic results (0, 1, 2)
8. `thalach`: Maximum heart rate achieved
9. `exang`: Exercise induced angina
10. `oldpeak`: ST depression induced by exercise relative to rest
11. `slope`: The slope of the peak exercise ST segment
12. `ca`: Number of major vessels (0-3) colored by fluoroscopy
13. `thal`: 3 = normal, 6 = fixed defect, 7 = reversible defect
14. `target`: 1 = presence of heart disease, 0 = absence of heart disease

## Installation
I have included requirements and dependencies files.

<p align="center">
  <img src="https://github.com/erenyeager101/ML-based-Heart-Disease-Detection/blob/main/pics/Q6_Image2_EDAI2.png" alt="Heart Disease Detection Result" width="600" />
</p>

# Usage
Clone the repository:
bash

git clone https://github.com/erenyeager101/Heart-Disease-monitoring.git
cd heart-disease-prediction
Ensure you have all dependencies installed.

Run the main script:

bash

python main.py

# Models Evaluated
The following models are evaluated in this project:

Logistic Regression
Naive Bayes
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
Neural Network
Results

base estimators =[ random forest , descision , knn]
The accuracy scores of the models are as follows:

Logistic Regression: 85.25%
Naive Bayes: 85.25%
Support Vector Machine: 81.97%
K-Nearest Neighbors: 67.21%
Decision Tree: 81.97%
Random Forest: 88.76%
Neural Network: 85.25%
Stacking Classifier: 90.16%

<p align="center">
  <img src="https://github.com/erenyeager101/ML-based-Heart-Disease-Detection/blob/main/pics/accuracy.png" alt="Accuracy" width="600" />
</p>

# Visualization
The project includes a bar plot that compares the accuracy scores of different models.



import matplotlib.pyplot as plt
import seaborn as sns

#Example code to plot the accuracy scores
plt.figure(figsize=(15, 8))
sns.barplot(x=algorithms, y=scores)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
plt.show()
Saving and Loading Models
The best model (Random Forest in this case, cause the accuracy we fetched using this model was highest in comparison to other algorithms )is saved using pickle for future use.


import pickle

# Save the model
with open('model_randomforestversion2.pkl', 'wb') as f:
    pickle.dump(rf, f)

    
# Contributing
Contributions are welcome! Please create a pull request or raise an issue to discuss your ideas.

# License
This project is licensed under the MIT License - see the LICENSE file for details.







