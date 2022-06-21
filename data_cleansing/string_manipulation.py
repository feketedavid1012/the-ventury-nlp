import re
import pandas as pd

from decision_making.task_selection import read_json


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
    if drop_outliers:
        classification_dataset = drop_outlier_categories(classification_dataset)
    for dic in classification_dataset:
        for key, vals in dic.items():
            if key != "id" and key != "score":
                if not stopping_word:
                    dic[key] = re.sub(r"[^a-zA-Z0-9']+", " ", vals).lower()
                else:
                    dic[key] = stopping_words(
                        re.sub(r"[^a-zA-Z0-9']+", " ", vals).lower())
            if key == "category":
                if dic[key] == "Blonde":
                    dic[key] = "Blond"
            sentence_length.append({"length": len(dic["body"])})

    if stopping_word:
        pd.DataFrame(classification_dataset).to_csv(
            "./data/cleaned_" + database_type + "dropped_outliers" + str(drop_outliers) + "_stopping_worded.csv")
        pd.DataFrame(sentence_length).describe().to_csv(
            "./data/statistics/sentence_length_" + database_type + "dropped_outliers" + str(drop_outliers) + "_stopping_worded.csv")
    else:
        pd.DataFrame(classification_dataset).to_csv(
            "./data/cleaned_" + database_type + "dropped_outliers" + str(drop_outliers) + ".csv")
        pd.DataFrame(sentence_length).describe().to_csv(
            "./data/statistics/sentence_length_" + database_type + "dropped_outliers" + str(drop_outliers) + ".csv")


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


def drop_outlier_categories(classification_dataset: str,important_features: list = [
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
])->str:
    """Drop categories which can be outliers

    Args:
        classification_dataset (str): Raw json data
        important_features (list, optional): Features, which are needed to be kept. Defaults to [ "Puns", "News / Politics", "Yo Momma", "Blond", "Insults", "At Work", "One Liners", "Animal", "Other / Misc", "Redneck", "Religious", "Men / Women", "Children", "Gross", "Medical" ].

    Returns:
        str: Json data which doesn't contain outlier categories.
    """
    data = pd.DataFrame(classification_dataset)
    data = data[data["category"].isin(important_features)]
    
    return data.to_dict()


if __name__ == "__main__":
    filter_strings(
        read_json("E:\\Austria\\the-ventury-nlp\\data\\raw\\classification.json"), False, True)
