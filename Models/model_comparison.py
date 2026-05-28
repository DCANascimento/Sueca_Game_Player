import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Number of games we want to use as the dataset
NUM_GAMES = 10000


# New top k accuracy metric
top3_scorer = make_scorer(
    top_k_accuracy_score, 
    k=3, 
    response_method='predict_proba',
    labels=list(range(40))
)

# Method for the way the model is evaluated
def eval_model(model, X_train, y_train):
    metrics = {
        "accuracy": "accuracy",
        "neg_log_loss": "neg_log_loss",
        "top3_accuracy": top3_scorer,
    }

    cv_results = cross_validate(
        model, X_train, y_train, cv=CV, scoring=metrics, n_jobs=-1, return_train_score=True
    )

    results = {
        "Train Accuracy": cv_results["train_accuracy"].mean(),
        "Val Accuracy": cv_results["test_accuracy"].mean(),
        "Val Accuracy Std": cv_results["test_accuracy"].std(),
        "Val Log Loss": -cv_results["test_neg_log_loss"].mean(),
        "Val Top-3 Accuracy": cv_results["test_top3_accuracy"].mean()
    }
    return results

# Method that converts the 40-bit vectors of the cards into separate columns
def expand_vector_columns(df, columns_to_expand):
    expanded_dfs = []
    columns_to_drop = []
    
    for col in columns_to_expand:
        if col in df.columns:
            print(f"Expanding 40-bit vector: {col}...")
            parsed_series = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
            # Create 40 distinct column headers for the matrix
            matrix = np.array(parsed_series.tolist())
            new_cols = [f"{col}_{i}" for i in range(matrix.shape[1])]
            
            expanded_dfs.append(pd.DataFrame(matrix, columns=new_cols, index=df.index))
            columns_to_drop.append(col)
            
    df = df.drop(columns=columns_to_drop)
    if expanded_dfs:
        df = pd.concat([df] + expanded_dfs, axis=1)
    return df

def plot_model_comparison(results_dict, filename="model_comparison.png"):
    model_names = list(results_dict.keys())

    val_acc = [results_dict[m]["Val Accuracy"] for m in model_names]
    log_loss = [results_dict[m]["Val Log Loss"] for m in model_names]
    top3 = [results_dict[m]["Val Top-3 Accuracy"] for m in model_names]

    y = np.arange(len(model_names))
    height = 0.25

    plt.figure(figsize=(10, 6))

    plt.barh(y - height, val_acc, height, label="Val Accuracy")
    plt.barh(y, top3, height, label="Top-3 Accuracy")
    plt.barh(y + height, log_loss, height, label="Log Loss")

    plt.yticks(y, model_names)
    plt.xlabel("Score")
    plt.title("Model Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved comparison plot to {filename}")

def main(path, target_column):
    print("Loading data...")
    df = pd.read_csv(path)

    df = df.head(40 * NUM_GAMES)


    # 1. Expand the 40-bit array lists into unique column features
    vector_cols = ['cards_on_table_snapshot', 'hand_before_play', 'legal_moves_play']
    df = expand_vector_columns(df, vector_cols)

    # 2. Prevent Data Leakage by removing identifier metadata columns
    # We drop game_number completely. We keep round_number, round_points, etc. as active features.
    columns_to_drop = [target_column, 'game_number']
    
    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    y = df[target_column]

    # Handle any potential missing fields gracefully
    X = X.fillna(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', C=10.0),
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            random_state=42, 
            learning_rate=0.03, 
            max_depth=None, 
            max_leaf_nodes=31, 
            min_samples_leaf=20
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob", 
            eval_metric="mlogloss", 
            random_state=42,
            colsample_bytree=0.8,
            learning_rate=0.05,
            max_depth=6,
            n_estimators=300,
            subsample=0.8,
            n_jobs=-1
        ),
    }

    results_summary = {}

    for name, model in models.items():
        print(f"\n=== Evaluating {name} via 5-Fold CV ===")
        metrics = eval_model(model, X_train, y_train)

        results_summary[name] = metrics

        for metric_name, value in metrics.items():
            if "Std" in metric_name:
                print(f"  {metric_name}: +/- {value:.4f}")
            else:
                print(f"  {metric_name}: {value:.4f}")

        print(f"Fitting final {name} model on full training subset...")
        model.fit(X_train, y_train)

        test_acc = model.score(X_test, y_test)
        print(f"  --> Final Unseen Holdout Test Accuracy (Top-1): {test_acc:.4f}")

        test_probabilities = model.predict_proba(X_test)
        test_top3_acc = top_k_accuracy_score(
            y_test, test_probabilities, k=3, labels=list(range(40))
        )
        print(f"  --> Final Unseen Holdout Top-3 Accuracy: {test_top3_acc:.4f}")
    plot_model_comparison(results_summary)


if __name__ == "__main__":
    # change this files when dataset is available
    dataset = "../Dataset/batch_rounds_flat_turns.csv" 
    target = "TARGET_card_played"

    main(dataset, target)