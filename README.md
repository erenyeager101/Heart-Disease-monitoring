# ❤️ Heart Disease Detection  
**Predict Heart Health with Hybrid Machine Learning Models**  

Heart disease detection using **hybrid machine learning** and **ensemble learning** models enables early diagnosis and treatment, potentially reducing mortality rates associated with cardiovascular diseases (CVD). This project demonstrates the power of data science in making healthcare more accessible and efficient. 🌍💡  

---

## 📝 **Research Paper**  
If you're interested in the research aspect of this project, explore the detailed paper:  
📄 [Read the Research Paper Here](http://dx.doi.org/10.13140/RG.2.2.15033.38247)  

---

## 🌐 **Live Demo**  
🚀 **Heart Disease Predictor Web App:** [Click Here](https://heart-disease-monitoring.onrender.com/)  
*⚠ Note: The site might take a few seconds to load due to server startup time.*  

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
The goal of this project is to predict the presence of heart disease in patients using machine learning models trained on clinical attributes. By leveraging ensemble learning, the system improves prediction accuracy and reliability.  

### 🛠 **Key Features:**  
- Hybrid machine learning models for improved accuracy.  
- Flask-based user-friendly web tool for remote accessibility.  
- Clean UI and easy-to-use prediction form.  
- Interactive charts and visual insights.  

---

## 📊 **Dataset Information**  
- **Dataset Used:** `heart.csv`  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)  

### **Attributes:**  
| Feature | Description |
|----------|-------------|
| age | Age of the patient |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic) |
| trestbps | Resting blood pressure (in mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| restecg | Resting electrocardiographic results (0, 1, 2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes; 0 = no) |
| oldpeak | ST depression induced by exercise relative to rest |
| slope | The slope of the peak exercise ST segment |
| ca | Number of major vessels (0–3) colored by fluoroscopy |
| thal | 3 = normal; 6 = fixed defect; 7 = reversible defect |
| target | 1 = presence of heart disease; 0 = absence of heart disease |

<p align="center">
  <img src="https://github.com/erenyeager101/ML-based-Heart-Disease-Detection/blob/main/pics/Q6_Image2_EDAI2.png" alt="Heart Disease Detection Result" width="600" />
</p>

---

## ⚙️ **Installation**  

### **Step 1:** Clone the repository  
```bash
git clone https://github.com/erenyeager101/Heart-Disease-monitoring.git
cd Heart-Disease-monitoring
```

### **Step 2:** Create and activate a virtual environment  
```bash
python -m venv venv
venv\Scripts\activate  # (Windows)
source venv/bin/activate  # (Mac/Linux)
```

### **Step 3:** Install dependencies  
```bash
pip install -r requirements.txt
```

### **Step 4:** Run the Flask web app  
```bash
python app.py
```

Now, open your browser and go to **http://127.0.0.1:5000/** 🚀  

---

## 💻 **Usage**  
- Open the web app.  
- Enter the required patient medical parameters.  
- Click **Predict** to see if the patient is likely to have heart disease.  
- The app displays the result instantly.  

---

## 🤖 **Models Evaluated**  
| Model | Accuracy |
|--------|-----------|
| Logistic Regression | 85.25% |
| Naive Bayes | 85.25% |
| Support Vector Machine (SVM) | 81.97% |
| K-Nearest Neighbors (KNN) | 67.21% |
| Decision Tree | 81.97% |
| Random Forest | 88.76% |
| Neural Network | 85.25% |
| **Stacking Classifier (Hybrid Model)** | **90.16%** |

> Base estimators used in Stacking Model: Random Forest, Decision Tree, and KNN.  

<p align="center">
  <img src="https://github.com/erenyeager101/ML-based-Heart-Disease-Detection/blob/main/pics/accuracy.png" alt="Model Accuracy Comparison" width="600" />
</p>

---

## 📈 **Visualization Example**  

```python
import matplotlib.pyplot as plt
import seaborn as sns

algorithms = ['LR', 'NB', 'SVM', 'KNN', 'DT', 'RF', 'NN', 'Stacking']
scores = [85.25, 85.25, 81.97, 67.21, 81.97, 88.76, 85.25, 90.16]

plt.figure(figsize=(12, 6))
sns.barplot(x=algorithms, y=scores)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy (%)")
plt.title("Model Performance Comparison")
plt.show()
```

---

## 💾 **Saving and Loading Models**  

```python
import pickle

# Save the model
with open('stacking_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model
with open('stacking_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

## 🤝 **Contributing**  
Contributions are welcome!  
- Fork the repository  
- Create a new branch (`feature-branch`)  
- Commit your changes  
- Open a pull request  

---

## 📜 **License**  
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.  

---

> Developed with ❤️ by [Kunal Sonne](https://github.com/erenyeager101)
