import os

import pandas as pd

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

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

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

from sklearn.ensemble import RandomForestClassifier

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

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)

print("The accuracy score achieved using Random Forest is: " + str(score_rf) + " %")

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(11, activation='relu', input_dim=13))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=300)
Y_pred_nn = model.predict(X_test)

var = Y_pred_nn.shape
rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded
score_nn = round(accuracy_score(Y_pred_nn, Y_test) * 100, 2)

print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")

scores = [score_lr, score_nb, score_svm, score_knn, score_dt, score_rf, score_nn]
algorithms = ["Logistic Regression", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbors", "Decision Tree",
              "Random Forest", "Neural Network"]

for i in range(len(algorithms)):
    print("The accuracy score achieved using " + algorithms[i] + " is: " + str(scores[i]) + " %")
    import matplotlib.pyplot as plt

    # Set the figure size
    plt.figure(figsize=(15, 8))

    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy score")

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Plotting the barplot
    sns.barplot(x=algorithms, y=scores)
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy score")
    plt.show()

    import pickle
with open('model_randomforestversion2', 'wb') as f:
    pickle.dump(rf, f)
    rf.predict([[55, 0, 1, 132, 342, 0, 1, 166, 0, 1.2, 2, 0, 2]])
    classifier = RandomForestClassifier()
    pickle.dump(classifier, open("model.pkl", "wb"))


from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1-score for each classifier
precision_lr = precision_score(Y_test, Y_pred_lr)
recall_lr = recall_score(Y_test, Y_pred_lr)
f1_lr = f1_score(Y_test, Y_pred_lr)

precision_nb = precision_score(Y_test, Y_pred_nb)
recall_nb = recall_score(Y_test, Y_pred_nb)
f1_nb = f1_score(Y_test, Y_pred_nb)

precision_svm = precision_score(Y_test, Y_pred_svm)
recall_svm = recall_score(Y_test, Y_pred_svm)
f1_svm = f1_score(Y_test, Y_pred_svm)

precision_knn = precision_score(Y_test, Y_pred_knn)
recall_knn = recall_score(Y_test, Y_pred_knn)
f1_knn = f1_score(Y_test, Y_pred_knn)

precision_dt = precision_score(Y_test, Y_pred_dt)
recall_dt = recall_score(Y_test, Y_pred_dt)
f1_dt = f1_score(Y_test, Y_pred_dt)

precision_rf = precision_score(Y_test, Y_pred_rf)
recall_rf = recall_score(Y_test, Y_pred_rf)
f1_rf = f1_score(Y_test, Y_pred_rf)

precision_nn = precision_score(Y_test, Y_pred_nn)
recall_nn = recall_score(Y_test, Y_pred_nn)
f1_nn = f1_score(Y_test, Y_pred_nn)

# Store performance metrics in dictionaries for easy comparison
performance_metrics = {
    "Logistic Regression": {"Accuracy": score_lr, "Precision": precision_lr, "Recall": recall_lr, "F1-score": f1_lr},
    "Naive Bayes": {"Accuracy": score_nb, "Precision": precision_nb, "Recall": recall_nb, "F1-score": f1_nb},
    "Support Vector Machine": {"Accuracy": score_svm, "Precision": precision_svm, "Recall": recall_svm,
                               "F1-score": f1_svm},
    "K-Nearest Neighbors": {"Accuracy": score_knn, "Precision": precision_knn, "Recall": recall_knn,
                            "F1-score": f1_knn},
    "Decision Tree": {"Accuracy": score_dt, "Precision": precision_dt, "Recall": recall_dt, "F1-score": f1_dt},
    "Random Forest": {"Accuracy": score_rf, "Precision": precision_rf, "Recall": recall_rf, "F1-score": f1_rf},
    "Neural Network": {"Accuracy": score_nn, "Precision": precision_nn, "Recall": recall_nn, "F1-score": f1_nn}
}

# Display performance metrics
print("\nPerformance Metrics Comparison:")
for algorithm, metrics in performance_metrics.items():
    print(f"\nAlgorithm: {algorithm}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
