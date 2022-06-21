from components.sequencer import NLP_Sequencer
import pandas as pd
import os
from keras_tuner import Hyperband
from components.tuner import NLPHyperModel

CREATE_TOKENIZER = False
FULL_PATH = "E:\\Austria\\the-ventury-nlp\\data\\cleaned\\cleaned_classification.csv"

CREATE_ENCODER = False

MAXLEN = 700
EPOCHS = 20
MAX_EPOCHS=3
PATIENCE=3

if CREATE_TOKENIZER:
    NLP_Sequencer.create_tokenizer(pd.read_csv(FULL_PATH)["body"].astype(str))

if CREATE_ENCODER:
    NLP_Sequencer.create_encoder(pd.read_csv(FULL_PATH)["category"].astype(str))
    
train_generator=NLP_Sequencer("E:\\Austria\\the-ventury-nlp\\data\\splits\\train_title_False.pickle", maxlen=MAXLEN, batch_size = 32)
validation_generator=NLP_Sequencer("E:\\Austria\\the-ventury-nlp\\data\\splits\\validation_title_False.pickle",maxlen=MAXLEN,batch_size = 32)

tuner = Hyperband(
            hypermodel=NLPHyperModel(epochs=EPOCHS, outp_shape=train_generator.output_shape, maxlen=MAXLEN, patience=PATIENCE),
            objective="val_accuracy",
            directory="E:\\Austria\\the-ventury-nlp\\data\\model\\tuner",
            max_epochs=MAX_EPOCHS,
            project_name="NLP_CLASSIFICATION"
        )
tuner.search(x=train_generator,validation_data=validation_generator,epochs=EPOCHS)

hypermodel = NLPHyperModel(epochs=EPOCHS, outp_shape=train_generator.output_shape, maxlen=MAXLEN, patience=PATIENCE)
best_hp = tuner.get_best_hyperparameters()[0]
model = hypermodel.build(best_hp)
train_history = hypermodel.fit(hp=best_hp,model=model,refit=True,x=train_generator,validation_data=validation_generator)


