from sklearn.neighbors import KNeighborsClassifier
from components.sequencer import NLP_Sequencer
import pandas as pd
import numpy as np
import os
from components.analyzer import save_confusion_matrix
from data_cleansing.preprocessing_and_splitting_cat import save_pickle


CREATE_TOKENIZER = True
FULL_PATH = "E:\\Austria\\the-ventury-nlp\\data\\cleaned\\cleaned_classification_dropped_outliers_False_stopping_worded.csv"

CREATE_ENCODER = True

MAXLEN = 1000
EPOCHS = 20
MAX_EPOCHS=8
PATIENCE=3

MODEL_PATH="E:\\Austria\\the-ventury-nlp\\data\\model\\NLP_CLASSIFICATION"
MODEL_FOLDER = "best_classfication_KN_stopping_worded"

if CREATE_TOKENIZER:
    NLP_Sequencer.create_tokenizer(pd.read_csv(FULL_PATH)["body"].astype(str))

if CREATE_ENCODER:
    NLP_Sequencer.create_encoder(pd.read_csv(FULL_PATH)["category"].astype(str))
    
train_generator=NLP_Sequencer("E:\\Austria\\the-ventury-nlp\\data\\splits\\train_title_False_stopping_worded_classification.pickle", maxlen=MAXLEN, batch_size = 32)
test_generator=NLP_Sequencer("E:\\Austria\\the-ventury-nlp\\data\\splits\\test_title_False_stopping_worded_classification.pickle",maxlen=MAXLEN,batch_size = 32)

model =  KNeighborsClassifier().fit(train_generator.transformed_inputs, train_generator.transformed_outputs)

predictions = model.predict(test_generator.transformed_inputs)

maxed_predictions = np.argmax(predictions,axis=1)
original_data = np.argmax(test_generator.transformed_outputs,axis=1)
save_confusion_matrix(original_data, maxed_predictions, os.path.join(MODEL_PATH,"confusion_matrix_classifier.png"))
save_pickle(os.path.join(MODEL_PATH,MODEL_FOLDER,"KN_model.pickle"),model)