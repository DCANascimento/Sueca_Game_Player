import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Here we mess with the metrics for model evaluation
def eval_model(model, X_train, y_train):
    metrics = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "neg_log_loss": "neg_log_loss",
    }

    cv_results = cross_validate(model, X_train, y_train, cv=CV, scoring=metrics, n_jobs=-1)

    results = {
        "accuracy": (
            cv_results["test_accuracy"].mean(),
            cv_results["test_accuracy"].std(),
        ),
        "f1_macro": (
            cv_results["test_f1_macro"].mean(),
            cv_results["test_f1_macro"].std(),
        ),
        "log_loss": (
            -cv_results["test_neg_log_loss"].mean(),
            cv_results["test_neg_log_loss"].std(),
        ),
    }

    return results


# Here we define the features, target and models (using the above given metrics)
def main(path, target_column):
    df = pd.read_csv(path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Models (wow)
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(multi_class="multinomial", max_iter=2000),
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss", random_state=42
        ),
    }

    # Here we test all the models and evaluate them based on the given metrics
    for name, model in models.items():
        print(f"\n=== Evaluating {name} via 5-Fold CV ===")
        cv_metrics = eval_model(model, X_train, y_train)

        for metric, (mean, std) in cv_metrics.items():
            print(f"{metric.capitalize()}: {mean:.4f} (+/- {std:.4f})")


        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print("Accuracy: " + test_acc)


if __name__ == "__main__":

    # We edit these 2 values to get the data + target
    dataset = "dataset.csv"
    target = "best card"

    main(dataset, target)