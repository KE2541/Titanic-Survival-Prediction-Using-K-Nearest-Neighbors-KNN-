
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load the dataset
data = pd.read_csv("tested.csv")

data.info()
print("\nMissing values:")
print(data.isnull().sum())


# 2. Fill missing ages
def fill_missing_ages(df):
    df = df.copy()

    # Use median age grouped by passenger class and sex
    df["Age"] = df["Age"].fillna(
        df.groupby(["Pclass", "Sex"])["Age"].transform("median")
    )

    # Fallback if any ages remain missing
    df["Age"] = df["Age"].fillna(df["Age"].median())

    return df


# 3. Data cleaning and feature engineering
def preprocess_data(data):
    df = data.copy()

    # Remove unnecessary columns
    df.drop(
        columns=["PassengerId", "Name", "Ticket", "Cabin"],
        errors="ignore",
        inplace=True
    )

    # Fill missing embarked values, then remove the column
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna("S")
        df.drop(columns=["Embarked"], inplace=True)

    # Fill missing ages
    df = fill_missing_ages(df)

    # Fill missing fares
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Convert gender to numeric
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    # Feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"]

    # 1 means traveling alone; 0 means with family
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)

    # Create fare and age bins
    df["FareBin"] = pd.qcut(
        df["Fare"].rank(method="first"),
        q=4,
        labels=False
    )

    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, np.inf],
        labels=False,
        include_lowest=True
    )

    return df


# 4. Preprocess the dataset
data = preprocess_data(data)

print("\nPreprocessed data:")
print(data.head())


# 5. Create features and target
if "Survived" not in data.columns:
    raise ValueError(
        "The dataset must contain a 'Survived' column "
        "for supervised training and evaluation."
    )

X = data.drop(columns=["Survived"])
y = data["Survived"]

# Handle any remaining missing feature values in the model pipeline
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)


# 6. Build a pipeline for preprocessing and KNN
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsClassifier())
])


# 7. Hyperparameter tuning
param_grid = {
    "knn__n_neighbors": range(1, 21),
    "knn__metric": ["euclidean", "manhattan", "minkowski"],
    "knn__weights": ["uniform", "distance"]
}


def tune_model(X_train, y_train):
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        f"Best cross-validation accuracy: "
        f"{grid_search.best_score_ * 100:.2f}%"
    )

    return grid_search.best_estimator_


# 8. Train and tune the model
best_model = tune_model(X_train, y_train)


# 9. Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Did Not Survive", "Survived"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(matrix)

    return predictions, accuracy, matrix


predictions, accuracy, matrix = evaluate_model(
    best_model,
    X_test,
    y_test
)


# 10. Plot the confusion matrix
def plot_model(matrix):
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Did Not Survive", "Survived"],
        yticklabels=["Did Not Survive", "Survived"]
    )

    plt.title("KNN Titanic Survival - Confusion Matrix")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.tight_layout()
    plt.show()


plot_model(matrix)




        