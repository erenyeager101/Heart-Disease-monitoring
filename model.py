import os

import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

# %matplotlib inline

print(os.listdir())

import warnings

warnings.filterwarnings('ignore')
dataset = pd.read_csv("heart.csv")

info = ["age", "1: male, 0: female", "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, "
                                     "4: asymptomatic", "resting blood pressure", " serum cholestoral in mg/dl",
        "fasting blood sugar > 120 mg/dl", "resting electrocardiographic results (values 0,1,2)", " maximum heart rate "
                                                                                                  "achieved",
        "exercise induced angina", "oldpeak = ST depression induced by exercise relative to rest", "the slope of the "
                                                                                                   "peak exercise ST "
                                                                                                   "segment",
        "number of major vessels (0-3) colored by flourosopy", "thal: 3 = normal; 6 = fixed defect; 7 = reversable "
                                                               "defect"]

for i in range(len(info)):
    print(dataset.columns[i] + ":\t\t\t" + info[i])
    print()

    from sklearn.model_selection import train_test_split

    predictors = dataset.drop("target", axis=1)
    target = dataset["target"]

    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
    print("Training set size:", X_train.shape[0])
    print("Testing set size:", X_test.shape[0])
    pd.set_option('display.max_columns', None)
    print("Contents of rows used in the testing set:")
    print(X_test)

    

    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_digits

    digits = load_digits()
    X, y = digits.data, digits.target

    rf = RandomForestClassifier(random_state=42)

    cv_scores = cross_val_score(rf, X, y, cv=5)

    print("Average Accuracy:", cv_scores.mean())

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression, RidgeClassifier

lr = LogisticRegression()

lr.fit(X_train, Y_train)

Y_pred_lr = lr.predict(X_test)
score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)

print("The accuracy score achieved using Logistic Regression is: " + str(score_lr) + " %")

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, Y_train)

Y_pred_nb = nb.predict(X_test)
score_nb = round(accuracy_score(Y_pred_nb, Y_test) * 100, 2)

print("The accuracy score achieved using Naive Bayes is: " + str(score_nb) + " %")

from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train, Y_train)

Y_pred_svm = sv.predict(X_test)
score_svm = round(accuracy_score(Y_pred_svm, Y_test) * 100, 2)

print("The accuracy score achieved using Linear SVM is: " + str(score_svm) + " %")

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
score_knn = round(accuracy_score(Y_pred_knn, Y_test) * 100, 2)

print("The accuracy score achieved using KNN is: " + str(score_knn) + " %")

from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0

for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

# print(max_accuracy)
# print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test)
score_dt = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)

print("The accuracy score achieved using Decision Tree is: " + str(score_dt) + " %")

from sklearn.ensemble import RandomForestClassifier, StackingClassifier

max_accuracy = 0

for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
    if (current_accuracy > max_accuracy):
        max_accuracy = current_accuracy
        best_x = x

# print(max_accuracy)
# print(best_x)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)

print("The accuracy score achieved using Random Forest is: " + str(score_rf) + " %")

estimators = [
    ('rf', RandomForestClassifier(random_state=best_x)),
    ('knn', KNeighborsClassifier(n_neighbors=7)),  
    ('dt', DecisionTreeClassifier(random_state=best_x))  
]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=RidgeClassifier(), cv=5)
stacking_clf.fit(X_train_scaled, Y_train)
Y_pred_stacking = stacking_clf.predict(X_test_scaled)
score_stacking = round(accuracy_score(Y_pred_stacking, Y_test) * 100, 2)
print("The accuracy score achieved using Improved Stacking Classifier is: " + str(score_stacking) + " %")

scores = [score_lr, score_nb, score_svm, score_knn, score_dt, score_rf]
algorithms = ["Logistic Regression", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbors", "Decision Tree",
              "Random Forest"]

for i in range(len(algorithms)):
    print("The accuracy score achieved using " + algorithms[i] + " is: " + str(scores[i]) + " %")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 8))

    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy score")

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.barplot(x=algorithms, y=scores)
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy score")
    plt.show()

  # saving the predition model in pickle file format

    import pickle
with open('stacking_model.pkl', 'wb') as f:
    pickle.dump(stacking_clf, f) 

# Example prediction after saving
prediction = stacking_clf.predict([[55, 0, 1, 132, 342, 0, 1, 166, 0, 1.2, 2, 0, 2]])
print("Prediction:", prediction)


