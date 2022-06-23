from components.sequencer import NLP_Sequencer
import pandas as pd
import scipy.stats
import os
from keras_tuner import Hyperband
from components.tuner import NLPHyperModel
from data_cleansing.preprocessing_and_splitting_cat import save_pickle, read_yaml
from components.analyzer import save_plot, plot_data, get_mse


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
    "E:\\Austria\\the-ventury-nlp\\data\\splits\\train_title_False_regression.pickle", maxlen=MAXLEN, batch_size=128)
validation_generator = NLP_Sequencer(
    "E:\\Austria\\the-ventury-nlp\\data\\splits\\validation_title_False_regression.pickle", maxlen=MAXLEN, batch_size=128)
test_generator = NLP_Sequencer(
    "E:\\Austria\\the-ventury-nlp\\data\\splits\\test_title_False_regression.pickle", maxlen=MAXLEN, batch_size=128)

hypermodel = NLPHyperModel(
    epochs=EPOCHS, outp_shape=train_generator.output_shape, maxlen=MAXLEN, patience=PATIENCE)
tuner = Hyperband(
    hypermodel=hypermodel,
    objective="val_loss",
    directory="E:\\Austria\\the-ventury-nlp\\data\\model\\tuner",
    max_epochs=MAX_EPOCHS,
    project_name="NLP_REGRESSION_500"
)
tuner.search(x=train_generator,
             validation_data=validation_generator, epochs=EPOCHS)

hypermodel = hypermodel
best_hp = tuner.get_best_hyperparameters()[0]
model = hypermodel.build(best_hp)
train_history = hypermodel.fit(hp=best_hp, model=model, refit=True,
                               x=train_generator, validation_data=validation_generator)

save_plot(MODEL_PATH, train_history.history, "loss")

evaluation = model.evaluate(test_generator)
save_pickle(os.path.join(MODEL_PATH, "evaluation.pickle"), evaluation)
predictions = model.predict(test_generator)
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
model.save(os.path.join(MODEL_PATH, MODEL_FOLDER))
