import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main(path, target_column):
    df = pd.read_csv(path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(multi_class="multinomial", max_iter=1000)

    model.fit(X_train, y_train)

    print("Accuracy:", model.score(X_test, y_test))


if __name__ == "__main__":

    # We edit these 2 values, and ONLY these 2 (unless model itself requires changes)
    dataset = ""
    target = ""

    main(dataset, target)