from components.sequencer import NLP_Sequencer
import pandas as pd
import numpy as np
import os
from keras_tuner import Hyperband
from components.tuner import NLPHyperModel
from components.analyzer import save_plot, save_confusion_matrix
from data_cleansing.preprocessing_and_splitting_cat import save_pickle


CREATE_TOKENIZER = True
FULL_PATH = "E:\\Austria\\the-ventury-nlp\\data\\cleaned\\cleaned_classification_dropped_outliers_True.csv"

CREATE_ENCODER = True

MAXLEN = 1000
EPOCHS = 20
MAX_EPOCHS=8
PATIENCE=3

MODEL_PATH="E:\\Austria\\the-ventury-nlp\\data\\model\\NLP_CLASSIFICATION"
MODEL_FOLDER = "best_classification_"

if CREATE_TOKENIZER:
    NLP_Sequencer.create_tokenizer(pd.read_csv(FULL_PATH)["body"].astype(str))

if CREATE_ENCODER:
    NLP_Sequencer.create_encoder(pd.read_csv(FULL_PATH)["category"].astype(str))
    
train_generator=NLP_Sequencer("E:\\Austria\\the-ventury-nlp\\data\\splits\\train_title_False.pickle", maxlen=MAXLEN, batch_size = 32)
validation_generator=NLP_Sequencer("E:\\Austria\\the-ventury-nlp\\data\\splits\\validation_title_False.pickle",maxlen=MAXLEN,batch_size = 32)
test_generator=NLP_Sequencer("E:\\Austria\\the-ventury-nlp\\data\\splits\\test_title_False.pickle",maxlen=MAXLEN,batch_size = 32)

hypermodel= NLPHyperModel(epochs=EPOCHS, outp_shape=train_generator.output_shape, maxlen=MAXLEN, patience=PATIENCE)
tuner = Hyperband(
            hypermodel= hypermodel,
            objective="val_accuracy",
            directory="E:\\Austria\\the-ventury-nlp\\data\\model\\tuner",
            max_epochs=MAX_EPOCHS,
            project_name="NLP_CLASSIFICATION"
        )
tuner.search(x=train_generator,validation_data=validation_generator,epochs=EPOCHS)

hypermodel = hypermodel
best_hp = tuner.get_best_hyperparameters()[0]
model = hypermodel.build(best_hp)
train_history = hypermodel.fit(hp=best_hp,model=model,refit=True,x=train_generator,validation_data=validation_generator)

save_plot(MODEL_PATH,train_history.history)
save_plot(MODEL_PATH,train_history.history,"loss")

evaluation = model.evaluate(test_generator)
save_pickle(os.path.join(MODEL_PATH,"evaluation.pickle"),evaluation)
predictions = model.predict(test_generator)

maxed_predictions = np.argmax(predictions,axis=1)
original_data = np.argmax(test_generator.transformed_outputs,axis=1)
save_confusion_matrix(original_data, maxed_predictions, os.path.join(MODEL_PATH,"confusion_matrix.png"))

model.save(os.path.join(MODEL_PATH,"model",MODEL_FOLDER))

