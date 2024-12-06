"""
============================================================
Filename: experiment.py
Author: Moritz Baumgart
Affiliation: University of Siegen, Intelligent Systems Group (ISG)
Date: December, 2024
============================================================

Description:
This file is part of the `knn` template for use with the AI scientist ('AI-S'), https://github.com/SakanaAI/AI-Scientist.
It implements a KNN algorithm for classification and the necessary auxiliary code (loading data, preprocessing, etc.) to enable AI-S to run the experiment.
============================================================
"""

from argparse import ArgumentParser
import json
from math import sqrt
from pathlib import Path
import pandas as pd


class KNN:
    """
    k-Nearest Neighbors (KNN) classifier.

    This class implements the KNN algorithm for classification. It uses
    Euclidean distance to identify the k nearest neighbors to a given test
    instance and predicts the class based on majority voting.
    """

    def __init__(self, num_neighbors) -> None:
        """
        Initialize the KNN classifier.

        Args:
            num_neighbors (int): The number of neighbors to consider for predictions.
        """
        self.num_neighbors = num_neighbors

        self.train: pd.DataFrame | None = None

    @staticmethod
    def euclidean_distance(v1, v2) -> float:
        """
        Calculate the Euclidean distance between two vectors.

        Args:
            v1 (list): The first vector.
            v2 (list): The second vector.

        Returns:
            float: The Euclidean distance between the two vectors.
        """

        distance = 0.0
        for i in range(len(v1) - 1):
            distance += (v1[i] - v2[i]) ** 2
        return sqrt(distance)

    def get_neighbors(self, test_row):
        """
        Locate the most similar neighbors for a given test instance.

        Args:
            test_row (pd.Series): A single test instance.

        Returns:
            list: A list of the k nearest neighbors from the training data.

        Raises:
            ValueError: If the training data has not been fitted using `fit`.
        """

        if self.train is None:
            raise ValueError("Train is none! Did you call fit first?")

        distances = []
        for _, train_row in self.train.iterrows():
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = []
        for i in range(self.num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    def fit(self, train: pd.DataFrame) -> None:
        """
        Save training data to be used for predictions.

        Args:
            train (pd.DataFrame): The training data.
        """
        self.train = train

    def predict(self, test: pd.DataFrame):
        """
        Make classification predictions for a test dataset.

        Args:
            test (pd.DataFrame): The test dataset with feature columns.

        Returns:
            list: A list of predicted class labels for the test instances.
        """

        predictions = []
        for _, test_row in test.iterrows():
            neighbors = self.get_neighbors(test_row)
            output_values = [row.iloc[len(row) - 1] for row in neighbors]
            predictions.append(max(set(output_values), key=output_values.count))
        return predictions


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.

    Accuracy is the ratio of correctly predicted instances to the total number
    of instances.

    Args:
        y_true (list): The ground truth labels.
        y_pred (list): The predicted labels.

    Returns:
        float: Accuracy score.
    """
    return sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)


def precision(y_true, y_pred):
    """
    Calculate the average precision across all classes.

    Precision is the ratio of true positives to the sum of true positives and
    false positives, calculated per class and averaged.

    Args:
        y_true (list): The ground truth labels.
        y_pred (list): The predicted labels.

    Returns:
        float: Precision score, averaged across classes.
    """

    classes = set(y_true)
    precision_scores = []
    for cls in classes:
        tp = sum((yt == cls) and (yp == cls) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != cls) and (yp == cls) for yt, yp in zip(y_true, y_pred))
        precision_scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return sum(precision_scores) / len(classes)


def recall(y_true, y_pred):
    """
    Calculate the average recall across all classes.

    Recall is the ratio of true positives to the sum of true positives and
    false negatives, calculated per class and averaged.

    Args:
        y_true (list): The ground truth labels.
        y_pred (list): The predicted labels.

    Returns:
        float: Recall score, averaged across classes.
    """

    classes = set(y_true)
    recall_scores = []
    for cls in classes:
        tp = sum((yt == cls) and (yp == cls) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == cls) and (yp != cls) for yt, yp in zip(y_true, y_pred))
        recall_scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return sum(recall_scores) / len(classes)


def f1_score(y_true, y_pred):
    """
    Calculate the F1 score, the harmonic mean of precision and recall.

    The F1 score is a measure of a model's accuracy on a dataset and combines
    precision and recall. It is computed as:
        F1 = (2 * precision * recall) / (precision + recall)

    Args:
        y_true (list): The ground truth labels.
        y_pred (list): The predicted labels.

    Returns:
        float: F1 score.
    """

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


def main():
    """
    Main function to run the experiment.

    This function sets up the argument parser, loads datasets, performs
    train-test splits, trains a KNN classifier, evaluates performance
    metrics, and saves the results to a JSON file.

    Command-Line Arguments:
        --out_dir (str): Output directory for the results. Default is "run_0".

    Output:
        A JSON file (`final_info.json`) containing performance metrics for each dataset.

    Raises:
        FileNotFoundError: If a dataset file cannot be found.
    """

    parser = ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    datasets = {"iris": "iris/iris.data", "wine": "wine/wine.csv"}

    final_info = {}

    for dataset_name, dataset_path in datasets.items():
        if args.out_dir == "run_0":
            data_path = Path(__file__).parent.parent.parent / "data" / dataset_path
        else:
            data_path = (
                Path(__file__).parent.parent.parent.parent / "data" / dataset_path
            )

        df = pd.read_csv(data_path, header=None)

        # Shuffle df and do a train test split
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        test_len = int(0.1 * df.shape[0])
        df_test = df.iloc[:test_len]
        true_labels = df_test[df_test.shape[1] - 1]  # Save labels separately
        df_test = df_test.drop(columns=df_test.shape[1] - 1)  # Drop labels from test df
        df_train = df.iloc[test_len:]

        knn = KNN(20)
        knn.fit(df_train)
        predictions = knn.predict(df_test)

        metrics = [accuracy, precision, recall, f1_score]

        results = {m.__name__: m(true_labels, predictions) for m in metrics}

        final_info[dataset_name] = {"means": results}

    out_dir = Path(__file__).parent / args.out_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(out_dir / "final_info.json", "w") as f:
        json.dump(final_info, f)


if __name__ == "__main__":
    main()
