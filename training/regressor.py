from sklearn.neighbors import KNeighborsRegressor
from components.sequencer import NLP_Sequencer
import pandas as pd
import os
import scipy.stats
from components.analyzer import plot_data, get_mse
from data_cleansing.preprocessing_and_splitting_cat import save_pickle, read_yaml

config = read_yaml(model_type="regression")
CREATE_TOKENIZER = config["CREATE_TOKENIZER"]
FULL_PATH = config["FULL_PATH"]

MAXLEN = config["MAXLEN"]
EPOCHS = config["EPOCHS"]
MAX_EPOCHS = config["MAX_EPOCHS"]
PATIENCE = config["PATIENCE"]

MODEL_PATH = config["MODEL_PATH"]
MODEL_FOLDER = config["MODEL_FOLDER"]

if CREATE_TOKENIZER:
    NLP_Sequencer.create_tokenizer(pd.read_csv(FULL_PATH)["body"].astype(str))

minmaxscaler = NLP_Sequencer.read_pickle(
    "E:\\Austria\\the-ventury-nlp\\data\\objects\\min_max_scaler.pickle")
powertransformer = NLP_Sequencer.read_pickle(
    "E:\\Austria\\the-ventury-nlp\\data\\objects\\power_transformer.pickle")

train_generator = NLP_Sequencer(
    "E:\\Austria\\the-ventury-nlp\\data\\splits\\train_title_False_regression.pickle", maxlen=MAXLEN, batch_size=32)
test_generator = NLP_Sequencer(
    "E:\\Austria\\the-ventury-nlp\\data\\splits\\test_title_False_regression.pickle", maxlen=MAXLEN, batch_size=32)

model = KNeighborsRegressor().fit(train_generator.transformed_inputs,
                                  train_generator.transformed_outputs)
predictions = model.predict(test_generator.transformed_inputs)

original_data = test_generator.transformed_outputs
original_inverted = minmaxscaler.inverse_transform(
    powertransformer.inverse_transform(original_data))
predicted_inverted = minmaxscaler.inverse_transform(
    powertransformer.inverse_transform(predictions))

pearsonr = scipy.stats.pearsonr(
    original_inverted[:, 0], predicted_inverted[:, 0])
spearman = scipy.stats.spearmanr(
    original_inverted[:, 0], predicted_inverted[:, 0])

plot_data(original_inverted, predicted_inverted, os.path.join(
    MODEL_PATH, MODEL_FOLDER, "first_100.png"))
mse = get_mse(original_inverted, predicted_inverted)
save_pickle(os.path.join(MODEL_PATH, MODEL_FOLDER, "mse.pickle"), mse)
save_pickle(os.path.join(MODEL_PATH, MODEL_FOLDER, "KN_model.pickle"), model)
