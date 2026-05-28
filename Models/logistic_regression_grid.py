import ast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
NUM_GAMES = 10000

top3_scorer = make_scorer(
    top_k_accuracy_score,
    k=3,
    response_method="predict_proba",
    labels=list(range(40)),
)


def expand_vector_columns(df, columns_to_expand):
    expanded_dfs = []
    columns_to_drop = []

    for col in columns_to_expand:
        if col in df.columns:
            print(f"Expanding 40-bit vector: {col}...")
            parsed_series = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            matrix = np.array(parsed_series.tolist())
            new_cols = [f"{col}_{i}" for i in range(matrix.shape[1])]
            expanded_dfs.append(pd.DataFrame(matrix, columns=new_cols, index=df.index))
            columns_to_drop.append(col)

    df = df.drop(columns=columns_to_drop)
    if expanded_dfs:
        df = pd.concat([df] + expanded_dfs, axis=1)
    return df


def main(path, target_column):
    print("Loading data...")
    df = pd.read_csv(path)
    df = df.head(40 * NUM_GAMES)

    vector_cols = ["cards_on_table_snapshot", "hand_before_play", "legal_moves_play"]
    df = expand_vector_columns(df, vector_cols)

    columns_to_drop = [target_column, "game_number"]
    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    y = df[target_column]
    X = X.fillna(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    param_grid = {
        "logisticregression__C": [0.1, 1.0, 10.0],
        "logisticregression__solver": ["lbfgs", "saga", "liblinear", "newton-cg", "newton-cholesky", "sag"],
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=CV,
        scoring={
            "accuracy": "accuracy",
            "neg_log_loss": "neg_log_loss",
            "top3_accuracy": top3_scorer,
        },
        refit="top3_accuracy",
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    best_index = search.best_index_
    cv_results = search.cv_results_

    print("\n=== Logistic Regression Grid Search Results ===")
    print(f"Best Params: {search.best_params_}")
    print(f"Train Accuracy: {cv_results['mean_train_accuracy'][best_index]:.4f}")
    print(f"Val Accuracy: {cv_results['mean_test_accuracy'][best_index]:.4f}")
    print(f"Val Accuracy Std: +/- {cv_results['std_test_accuracy'][best_index]:.4f}")
    print(f"Val Log Loss: {-cv_results['mean_test_neg_log_loss'][best_index]:.4f}")
    print(f"Val Top-3 Accuracy: {cv_results['mean_test_top3_accuracy'][best_index]:.4f}")

    best_model = search.best_estimator_
    print("Fitting final Logistic Regression model on full training subset...")
    best_model.fit(X_train, y_train)

    test_acc = best_model.score(X_test, y_test)
    print(f"  --> Final Unseen Holdout Test Accuracy (Top-1): {test_acc:.4f}")

    test_probabilities = best_model.predict_proba(X_test)
    test_top3_acc = top_k_accuracy_score(y_test, test_probabilities, k=3, labels=list(range(40)))
    print(f"  --> Final Unseen Holdout Top-3 Accuracy: {test_top3_acc:.4f}")


if __name__ == "__main__":
    dataset = "../Dataset/batch_rounds_flat_turns.csv"
    target = "TARGET_card_played"
    main(dataset, target)