import json
import pandas as pd


def analyze_classification(classification_dataset: str, filepath: str = "./decision_making/classification_results.json") -> str:
    """Statistical metrics about classification task.

    Args:
        classification_dataset (str): Classification database in json format.
        filepath (str, optional): Destination path, where to save statistics. Defaults to "./decision_making/classification_results.json".

    Returns:
        str: Statistics as dictionary.
    """
    length = len(classification_dataset)
    labels = [dic["category"] for dic in classification_dataset]
    nb_of_labels = len(set(labels))
    occurance_of_labels = [(i, labels.count(i)) for i in set(labels)]
    category_in_title = len(set(
        dic["id"] for dic in classification_dataset if dic["category"].lower() in dic["title"].lower()))
    statistics = {"number_of_datapoints": length, "number_of_labels": nb_of_labels,
                  "occurance_of_labels": occurance_of_labels, "category_in_title": category_in_title}
    statistics["not_outlier_categories"] = cls_non_outliers(statistics)
    statistics["number_of_non_outliers"] = len(
        statistics["not_outlier_categories"])
    with open(filepath, "w") as file:
        json.dump(statistics, file, indent=4)

    return statistics


def analyze_regression(regression_dataset: str, filepath: str = "./decision_making/regression_results.json") -> str:
    """Statistical metrics about regression task.

    Args:
        regression_dataset (str): Regression database in json format.
        filepath (str, optional): Destination path, where to save statistics. Defaults to "./decision_making/regression_results.json".

    Returns:
        str: Statistics as dictionary.
    """
    length = len(regression_dataset)
    scores = pd.DataFrame([dic["score"] for dic in regression_dataset])
    description = scores.describe()
    statistics = {"number_of_datapoints": length,
                  "data_statistic": description.to_json()}
    with open(filepath, "w") as file:
        json.dump(statistics, file, indent=4)
    return statistics


def cls_non_outliers(statistics: str, percentage: float = 0.02) -> list:
    """Get the non-outlier categories from labels

    Args:
        statistics (str): Statistics of classification data
        percentage (float, optional): Outlier threshold. It's multiplied by the number of datapoints. Defaults to 0.02.

    Returns:
        list: Labels which are not outliers
    """
    outlier_threshold = statistics["number_of_datapoints"]*percentage

    non_outliers = [val[0] for idx, val in enumerate(
        statistics["occurance_of_labels"]) if val[1] > outlier_threshold]

    return non_outliers
