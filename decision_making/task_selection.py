import json
from decision_making.analyzation import analyze_regression, analyze_classification


def read_json(filepath: str):
    """Read json file

    Args:
        filepath (str): Filepath to json database

    Returns:
        str: Return readed json object
    """
    dataset = open(filepath)
    return json.load(dataset)


def run_analyze(cls_filepath: str = "./data/raw/classification.json", reg_filepath: str = "./data/raw/regression.json"):
    """Generate statistics about databases

    Args:
        cls_filepath (str, optional): Filepath to classification json database. Defaults to "./data/classification.json".
        reg_filepath (str, optional): Filepath to regression json database. Defaults to "./data/regression.json".

    Returns:
        tuple: Statistics about classification and regression.
    """
    classification_dataset = read_json(cls_filepath)
    regression_dataset = read_json(reg_filepath)

    cls_statistics = analyze_classification(classification_dataset)
    reg_statistics = analyze_regression(regression_dataset)

    return cls_statistics, reg_statistics


if __name__ == "__main__":

    cls_statistics, reg_statistics = run_analyze()

    """
    Classification result:
        - Enough data (over 10k)
        - There are 24 labels, but some of them have less then 2% occurence, thus they can handled as outliers.
        - If they are handled as outliers there will remain only 15 categories
        - In laboratory conditions the title could be used during the training, because 607 times the category shows up in the title. In practice, I wouldn't use it.
        - There is a "Blond" and a "Blonde" category, I would merge them, it looks like a typo. Same with Yo Mama and Yo Moma.
        - I would undersample categories which has 900 + occurance
        
        Preprocessing steps:
        - Data filtering based on label outliers
        - Cleaning strings from non understandable chars, substrings such as \n, $ sign.
        - Stopping word cleaning also could be used e.g. he, she, don't, but in case of "Blond" category it could be useful
        - Tokenizing words
    
    Regression result:
        - Enough data (ove 190k)
        - There are outliers in results: I would use 68-95-99.7 rule
        
        Preprocessing steps:
            - String:
                - Cleaning strings from non understandable chars, substrings such as \n, $ sign.
                - Stopping word cleaning also could be used e.g. he, she, don't, but in case of "Blond" category it could be useful
                - Tokenizing words
            - Numbers:
                - Use 68-95-99.7 rule, drop the given records
                - The distribution of data is kind of Lognormal, I would use PowerTransformer, and MinMaxScaler
    """
