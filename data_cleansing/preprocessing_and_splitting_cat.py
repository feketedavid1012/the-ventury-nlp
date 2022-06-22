import os
import pickle
import pandas as pd
from typing import Any
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def create_splits(data_path: str = "E:\\Austria\\the-ventury-nlp\\data\\cleaned\\cleaned_classification.csv", path_to_save: str = "E:\\Austria\\the-ventury-nlp\\data\\splits", title_needed: bool = False):
    """Creates train-test-split and saves it to pickle .

    Args:
        data_path (str, optional): Path to cleaned data. Defaults to "E:\\Austria\\the-ventury-nlp\\data\\cleaned_classification.csv".
        path_to_save (str, optional): Path where splits will be saved. Defaults to "E:\\Austria\\the-ventury-nlp\\data\\splits".
        title_needed (bool, optional): If True title will be concated to body and category. Defaults to False.
    """
    data = pd.read_csv(data_path)
    data = shuffle(data)
    categorical = pd.DataFrame(data["category"],columns=["category"])
    if title_needed:
        inputs = data[["body", "title"]]
    else:
        inputs = pd.DataFrame(data["body"],columns=["body"])
    body_train, body_test, category_train, category_test = splitting(
        inputs, categorical, test_size=0.3, random_state=42)
    body_validation, body_test, category_validation, category_test = splitting(
        body_test, category_test, test_size=0.5, random_state=42)

    train_set=pd.concat([body_train,category_train],axis=1)
    validation_set=pd.concat([body_validation,category_validation],axis=1)
    test_set=pd.concat([body_test,category_test],axis=1)
    
    datas=[train_set,validation_set,test_set]
    splits = ["train", "validation", "test"]
    for idx, val in enumerate(splits):
        if "stopping_worded" in data_path:
            filepath = os.path.join(path_to_save, val + "_title_" + str(title_needed)+ "_stopping_worded_" + ".pickle")
        else:
            filepath = os.path.join(path_to_save, val + "_title_" + str(title_needed) + ".pickle")
        save_pickle(filepath, datas[idx])

def save_pickle(filepath: str, variable_to_save: Any):
    """Saving variables to pickle

    Args:
        filepath (str): Path to save
        variable_to_save (Any): Variable to save
    """    
    with open(filepath, 'wb') as handle:
            pickle.dump(variable_to_save, handle,
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
    create_splits("E:\\Austria\\the-ventury-nlp\\data\\cleaned\\cleaned_classification_dropped_outliers_True.csv")
