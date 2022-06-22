import re
import json
import pandas as pd
import numpy as np

from decision_making.task_selection import read_json
from data_cleansing.preprocessing_and_splitting_cat import save_pickle


def filter_strings(classification_dataset: str, stopping_word: bool = False, database_type: str = "classification", drop_outliers: bool = False):
    """Drop non alphanumeric and "'" characters and save it to csv.
    Replace "Blonde" to "Blond". If stopping_word True, drop the stopping words.

    Args:
        classification_dataset (str): Classification json database
        stopping_word (bool, optional): Stopping word filter is needed, or not. Defaults to False.
        database_type (str, optional): Type of database. Classification or regression. Defaults to "classification".
        drop_outliers (bool, optional): If True it will drop outlier categories. Defaults to False.
    """
    sentence_length = []

    if database_type == "classification":
        if drop_outliers:
            classification_dataset = drop_outlier_categories(
                classification_dataset)
        classification_dataset = undersampling(classification_dataset)
    elif database_type == "regression":
        dataset = pd.DataFrame(classification_dataset)
        #dataset = dataset[dataset["score"]<(dataset["score"].mean()+0.5*dataset["score"].std())]
        dataset["score"] = powertransforming(
            dataset["score"].values.reshape(-1, 1))
        dataset["score"] = minmaxscaling(
            dataset["score"].values.reshape(-1, 1))
        classification_dataset = df_to_json(dataset)
        
    for dic in classification_dataset:
        for key, vals in dic.items():
            if key != "id" and key != "score":
                if not stopping_word:
                    dic[key] = re.sub(r"[^a-zA-Z0-9']+", " ", vals).lower()
                else:
                    dic[key] = stopping_words(
                        re.sub(r"[^a-zA-Z0-9']+", " ", vals).lower())

                if key == "category":
                    if dic[key] == "blonde":
                        dic[key] = "blond"
                    elif dic[key] == "yo mama":
                        dic[key] = "yo momma"

            sentence_length.append({"length": len(dic["body"])})

    if stopping_word:
        pd.DataFrame(classification_dataset).to_csv(
            "./data/cleaned/cleaned_" + database_type + "_dropped_outliers_" + str(drop_outliers) + "_stopping_worded.csv")
        pd.DataFrame(sentence_length).describe().to_csv(
            "./data/statistics/sentence_length_" + database_type + "_dropped_outliers_" + str(drop_outliers) + "_stopping_worded.csv")
    else:
        pd.DataFrame(classification_dataset).to_csv(
            "./data/cleaned/cleaned_" + database_type + "_dropped_outliers_" + str(drop_outliers) + ".csv")
        pd.DataFrame(sentence_length).describe().to_csv(
            "./data/statistics/sentence_length_" + database_type + "_dropped_outliers_" + str(drop_outliers) + ".csv")


def minmaxscaling(data: np.ndarray) -> np.ndarray:
    """Scaling values between 0-1.

    Args:
        data (np.ndarray): Not scaler array.

    Returns:
        np.ndarray: Scaled array.
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    save_pickle(
        "E:\\Austria\\the-ventury-nlp\\data\\objects\\min_max_scaler.pickle", scaler)
    return data


def powertransforming(data: np.ndarray) -> np.ndarray:
    """Power transform to make data Gaussian.

    Args:
        data (np.ndarray): Not Gaussian-like data.

    Returns:
        np.ndarray: Transformed data.
    """
    from sklearn.preprocessing import PowerTransformer
    scaler = PowerTransformer()
    data = scaler.fit_transform(data)
    save_pickle(
        "E:\\Austria\\the-ventury-nlp\\data\\objects\\power_transformer.pickle", scaler)
    return data


def stopping_words(sentence: str) -> str:
    """Filter the stopping word from a sentence, based on nltk's stopping word collection.

    Args:
        sentence (str): Input sentence with stopping words.

    Returns:
        str: Filtered sentence without stopping words.
    """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    return " ".join(filtered_sentence)


def drop_outlier_categories(classification_dataset: str, important_features: list = [
    "Puns",
    "News / Politics",
    "Yo Momma",
    "Blond",
    "Insults",
    "At Work",
    "One Liners",
    "Animal",
    "Other / Misc",
    "Redneck",
    "Religious",
    "Men / Women",
    "Children",
    "Gross",
    "Medical"
]) -> str:

    """Drop categories which can be outliers

    Args:
        classification_dataset (str): Raw json data
        important_features (list, optional): Features, which are needed to be kept. Defaults to [ "Puns", "News / Politics", "Yo Momma", "Blond", "Insults", "At Work", "One Liners", "Animal", "Other / Misc", "Redneck", "Religious", "Men / Women", "Children", "Gross", "Medical" ].

    Returns:
        str: Json data which doesn't contain outlier categories.
    """
    data = pd.DataFrame(classification_dataset)
    data = data[data["category"].isin(important_features)]

    return df_to_json(data)


def undersampling(classification_dataset: str, feature: list = ["Other / Misc", "Men / Women", "One Liners"]):

    data = pd.DataFrame(classification_dataset)
    feature_set = []
    for idx, val in enumerate(feature):
        feature_set.append(data[data["category"].isin([val])].sample(600))
    feature_set.append(data[~data["category"].isin(feature)])
    data = pd.concat(feature_set)

    return df_to_json(data)


def df_to_json(data: pd.DataFrame) -> str:
    return json.loads(data.to_json(orient='records'))


if __name__ == "__main__":
    filter_strings(
        read_json("E:\\Austria\\the-ventury-nlp\\data\\raw\\regression.json"), True, database_type="regression")
