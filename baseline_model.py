import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import argparse


def main(train_path: str, test_path: str, output_path: str) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Assume the target column is named 'target'. Adjust if competition uses a different label.
    y = train_df['target']
    X = train_df.drop(columns=['target', 'id'], errors='ignore')

    model = RandomForestClassifier(n_estimators=200, random_state=42)

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    model.fit(X, y)

    X_test = test_df.drop(columns=['id'], errors='ignore')
    preds = model.predict(X_test)
    sub = pd.DataFrame({'id': test_df.get('id', range(len(preds))), 'target': preds})
    sub.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline model for Kaggle PS5E6')
    parser.add_argument('--train', default='train.csv', help='Path to train.csv')
    parser.add_argument('--test', default='test.csv', help='Path to test.csv')
    parser.add_argument('--output', default='submission.csv', help='Output CSV file')
    args = parser.parse_args()
    main(args.train, args.test, args.output)
