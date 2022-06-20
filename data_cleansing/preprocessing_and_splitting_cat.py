import os
import pickle
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def one_hot_encoding(data_path: str = "E:\\Austria\\the-ventury-nlp\\data\\cleaned_classification.csv", path_to_save: str = "E:\\Austria\\the-ventury-nlp\\data\\splits", title_needed: bool = False):
    """Create One-Hot-Encoded data from categorical values.

    Args:
        data_path (str, optional): Path to cleaned data. Defaults to "E:\\Austria\\the-ventury-nlp\\data\\cleaned_classification.csv".
        path_to_save (str, optional): Path where splits will be saved. Defaults to "E:\\Austria\\the-ventury-nlp\\data\\splits".
        title_needed (bool, optional): If True title will be concated to body and category. Defaults to False.
    """
    data = pd.read_csv(data_path)
    data = shuffle(data)
    categorical = data["category"]
    categorical = pd.get_dummies(categorical)
    if title_needed:
        inputs = data[["body", "title"]]
    else:
        inputs = data["body"]
    body_train, body_test, category_train, category_test = splitting(
        inputs, categorical, test_size=0.3, random_state=42)
    body_validation, body_test, category_validation, category_test = splitting(
        body_test, category_test, test_size=0.5, random_state=42)

    datas = [body_train, category_train, body_validation,
             category_validation, body_test, category_test]
    splits = ["train", "validation", "test"]
    for idx, val in enumerate(splits):
        with open(os.path.join(path_to_save, val + "_title_" + str(title_needed) + ".pickle"), 'wb') as handle:
            pickle.dump(datas[idx*2:((idx+1)*2)], handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


def splitting(inputs: pd.DataFrame, outputs: pd.DataFrame, *args, **kwargs) -> tuple:
    """Splitting data into to parts.

    Args:
        inputs (array-like): Input data.
        outputs (array-like): Output data.

    Returns:
        tuple: Splitted data: (input_1, input_2, output_1, output_2) format.
    """
    body_train, body_test, category_train, category_test = train_test_split(
        inputs, outputs, *args, **kwargs)
    return body_train, body_test, category_train, category_test


if __name__ == "__main__":
    one_hot_encoding()
