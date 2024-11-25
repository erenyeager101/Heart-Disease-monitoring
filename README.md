# ❤️ Heart Disease Detection  
**Predict Heart Health with Hybrid Machine Learning Models**  

Heart disease detection using **hybrid machine learning** and **ensemble learning** models enables early diagnosis and treatment, potentially reducing mortality rates associated with cardiovascular diseases (CVD). This project demonstrates the power of data science in making healthcare more accessible and efficient. 🌍💡  

---

## 📝 **Research Done**  
If you're interested in the research aspect of this project, explore my detailed paper:  
📄 [Read the Research Paper Here](http://dx.doi.org/10.13140/RG.2.2.15033.38247)  

---

## 🌐 **Heart Disease Prediction**  
This project involves building and evaluating machine learning models to predict heart disease based on the `heart.csv` dataset. The tool is deployed online, so anyone can access it for predictions!  

🚀 **Live Demo**: [Heart Disease Predictor](https://heart-disease-monitoring.onrender.com/)  
*⚠ Note: The site might take a few seconds to load due to server traffic.*  

---

## 📚 **Table of Contents**  
- [Overview](#overview)  
- [Dataset Information](#dataset-information)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Models Evaluated](#models-evaluated)  
- [Results](#results)  
- [Visualization](#visualization)  
- [Contributing](#contributing)  
- [License](#license)  

---

## 🎥 **Video Overview**  
🔗 [Watch the Project Walkthrough](https://youtu.be/3Txp134133s)  

[![Machine Learning Based Heart Disease Detection](https://img.youtube.com/vi/3Txp134133s/0.jpg)](https://www.youtube.com/watch?v=3Txp134133s)  

---

## 🌟 **Overview**  
The goal of this project is to predict the presence of heart disease in patients using machine learning models trained on medical attributes. By leveraging ensemble learning, we improve accuracy and reliability.  

🛠 **Key Features:**  
- Hybrid machine learning models for better accuracy.  
- User-friendly web-based tool for remote accessibility.  
- Rich data visualizations for insights.  

---

## 📊 **Dataset Information**  
- **Dataset Used:** `heart.csv`  
- **Attributes:**  
  - Age, Sex, Chest Pain Type, Resting BP, Serum Cholesterol, Fasting Blood Sugar, Resting ECG, Max HR, Exercise-induced Angina, Oldpeak, Slope, Number of Vessels, and Thal.  
- **Source:**


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







