import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train():
    mlflow.set_experiment("iris_training")
    iris = load_iris()

    iris = pd.DataFrame(
        np.c_[iris["data"], iris["target"]], columns=iris.feature_names + ["target"]
    )
    iris.head()
    species = []

    for i in range(len(iris["target"])):
        if iris["target"][i] == 0:
            species.append("setosa")
        elif iris["target"][i] == 1:
            species.append("versicolor")
        else:
            species.append("virginica")

    iris["species"] = species

    # Droping the target and species since we only need the measurements
    X = iris.drop(["target", "species"], axis=1)

    # converting into numpy array and assigning petal length and petal width
    X = X.to_numpy()[:, (2, 3)]
    y = iris["target"]

    # Splitting into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log_reg = LogisticRegression()
    with mlflow.start_run(run_name="iris_training"):
        mlflow.log_param("random_state", 42)
        log_reg.fit(X_train, y_train)
        # add metrics to the run
        mlflow.log_metric("train_accuracy", log_reg.score(X_train, y_train))
        mlflow.log_metric("test_accuracy", log_reg.score(X_test, y_test))
        mlflow.log_param("target", y.name)
        mlflow.sklearn.log_model(
            log_reg, name="iris_model", registered_model_name="iris_model"
        )


if __name__ == "__main__":
    train()
