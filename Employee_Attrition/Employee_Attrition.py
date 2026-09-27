import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("Employee_Attrition.csv")


# ============================================================
# 2. DISPLAY SHAPE, COLUMNS AND FIRST 5 RECORDS
# ============================================================

print("Shape of Dataset:")
print(data.shape)

print("\nColumns:")
print(data.columns)

print("\nFirst 5 Records:")
print(data.head())


# ============================================================
# 3. CHECK FOR MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(data.isnull().sum())


# ============================================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "TotalWorkingYears",
    "DistanceFromHome",
    "JobSatisfaction",
    "WorkLifeBalance",
    "NumCompaniesWorked",
    "TrainingTimesLastYear"
]

categorical_features = [
    "OverTime"
]

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 5. CONVERT CATEGORICAL FEATURE INTO NUMERICAL
# ============================================================

data["OverTime"] = data["OverTime"].map({
    "Yes": 1,
    "No": 0
})


# ============================================================
# 6. CONVERT TARGET ATTRITION INTO 0 AND 1
# ============================================================

data["Attrition"] = data["Attrition"].map({
    "Yes": 1,
    "No": 0
})

print("\nData After Encoding:")
print(data.head())


# ============================================================
# 7. SEPARATE INDEPENDENT AND DEPENDENT VARIABLES
# ============================================================

X = data.drop("Attrition", axis=1)
y = data["Attrition"]


print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)


# ============================================================
# 8. DIVIDE DATA INTO TRAINING AND TESTING DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# 9. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 10. DESIGN MLP WITH AT LEAST TWO HIDDEN LAYERS
# ============================================================

model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)


# ============================================================
# 11. TRAIN THE NETWORK
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 12. DISPLAY NUMBER OF ITERATIONS
# ============================================================

print("\nNumber of Iterations:")
print(model.n_iter_)


# ============================================================
# 13. CALCULATE TRAINING ACCURACY
# ============================================================

y_train_pred = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

print("\nTraining Accuracy:")
print(train_accuracy)


# ============================================================
# 14. CALCULATE TESTING ACCURACY
# ============================================================

y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print("\nTesting Accuracy:")
print(test_accuracy)


# ============================================================
# 15. GENERATE CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stay", "Leave"]
)

disp.plot()

plt.title("Employee Attrition Confusion Matrix")
plt.show()


# ============================================================
# 16. PLOT LOSS CURVE
# ============================================================

plt.plot(
    model.loss_curve_
)

plt.title("MLP Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.show()


# ============================================================
# 17. PREDICT ATTRITION FUNCTION
# ============================================================

def PredictAttrition(employee_data):

    employee_data = pd.DataFrame(
        [employee_data]
    )

    # Convert OverTime
    employee_data["OverTime"] = employee_data["OverTime"].map({
        "Yes": 1,
        "No": 0
    })

    # Scale input data
    employee_data = scaler.transform(employee_data)

    # Prediction
    prediction = model.predict(employee_data)

    if prediction[0] == 1:
        return 1
    else:
        return 0


# ============================================================
# 18. TEST USING FIVE NEW EMPLOYEE RECORDS
# ============================================================

employees = [
    {
        "Age": 26,
        "MonthlyIncome": 3200,
        "YearsAtCompany": 1,
        "TotalWorkingYears": 3,
        "DistanceFromHome": 15,
        "JobSatisfaction": 2,
        "WorkLifeBalance": 2,
        "OverTime": "Yes",
        "NumCompaniesWorked": 3,
        "TrainingTimesLastYear": 2
    },

    {
        "Age": 40,
        "MonthlyIncome": 7500,
        "YearsAtCompany": 10,
        "TotalWorkingYears": 17,
        "DistanceFromHome": 5,
        "JobSatisfaction": 4,
        "WorkLifeBalance": 4,
        "OverTime": "No",
        "NumCompaniesWorked": 2,
        "TrainingTimesLastYear": 4
    },

    {
        "Age": 30,
        "MonthlyIncome": 4200,
        "YearsAtCompany": 3,
        "TotalWorkingYears": 7,
        "DistanceFromHome": 12,
        "JobSatisfaction": 2,
        "WorkLifeBalance": 2,
        "OverTime": "Yes",
        "NumCompaniesWorked": 4,
        "TrainingTimesLastYear": 2
    },

    {
        "Age": 37,
        "MonthlyIncome": 6500,
        "YearsAtCompany": 8,
        "TotalWorkingYears": 14,
        "DistanceFromHome": 6,
        "JobSatisfaction": 4,
        "WorkLifeBalance": 3,
        "OverTime": "No",
        "NumCompaniesWorked": 2,
        "TrainingTimesLastYear": 3
    },

    {
        "Age": 28,
        "MonthlyIncome": 3600,
        "YearsAtCompany": 2,
        "TotalWorkingYears": 5,
        "DistanceFromHome": 18,
        "JobSatisfaction": 2,
        "WorkLifeBalance": 2,
        "OverTime": "Yes",
        "NumCompaniesWorked": 3,
        "TrainingTimesLastYear": 1
    }
]


print("\nPredictions for New Employees:")

for i, employee in enumerate(employees, start=1):

    prediction = PredictAttrition(employee)

    print("\nEmployee", i)

    if prediction == 1:
        print("Prediction: 1 - Employee is likely to leave")
    else:
        print("Prediction: 0 - Employee is likely to stay")


# ============================================================
# 19. CHECK OVERFITTING / UNDERFITTING
# ============================================================

print("\nModel Analysis:")

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

if train_accuracy > test_accuracy + 0.10:
    print("The model may be suffering from overfitting.")

elif train_accuracy < 0.70 and test_accuracy < 0.70:
    print("The model may be suffering from underfitting.")

else:
    print("There is no strong indication of overfitting or underfitting.")

