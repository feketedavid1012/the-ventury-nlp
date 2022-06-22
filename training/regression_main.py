from components.sequencer import NLP_Sequencer
import pandas as pd
import scipy.stats
import os
from keras_tuner import Hyperband
from components.tuner import NLPHyperModel
from data_cleansing.preprocessing_and_splitting_cat import save_pickle
from components.analyzer import save_plot, plot_data


CREATE_TOKENIZER = True
FULL_PATH = "E:\\Austria\\the-ventury-nlp\\data\\cleaned\\cleaned_regression_dropped_outliers_False_stopping_worded.csv"

MAXLEN = 1000
EPOCHS = 20
MAX_EPOCHS=3
PATIENCE=2

MODEL_PATH="E:\\Austria\\the-ventury-nlp\\data\\model\\NLP_REGRESSION"
MODEL_FOLDER = "best_regression"

if CREATE_TOKENIZER:
    NLP_Sequencer.create_tokenizer(pd.read_csv(FULL_PATH)["body"].astype(str))
    
minmaxscaler = NLP_Sequencer.read_pickle("E:\\Austria\\the-ventury-nlp\\data\\objects\\min_max_scaler.pickle")
powertransformer = NLP_Sequencer.read_pickle("E:\\Austria\\the-ventury-nlp\\data\\objects\\power_transformer.pickle")
    
train_generator=NLP_Sequencer("E:\\Austria\\the-ventury-nlp\\data\\splits\\train_title_False_stopping_worded_regression.pickle", maxlen=MAXLEN, batch_size = 128)
validation_generator=NLP_Sequencer("E:\\Austria\\the-ventury-nlp\\data\\splits\\validation_title_False_stopping_worded_regression.pickle",maxlen=MAXLEN,batch_size = 128)
test_generator=NLP_Sequencer("E:\\Austria\\the-ventury-nlp\\data\\splits\\test_title_False_stopping_worded_regression.pickle",maxlen=MAXLEN,batch_size = 128)

hypermodel= NLPHyperModel(epochs=EPOCHS, outp_shape=train_generator.output_shape, maxlen=MAXLEN, patience=PATIENCE)
tuner = Hyperband(
            hypermodel= hypermodel,
            objective="val_loss",
            directory="E:\\Austria\\the-ventury-nlp\\data\\model\\tuner",
            max_epochs=MAX_EPOCHS,
            project_name="NLP_REGRESSION_stopping_worded"
        )
tuner.search(x=train_generator,validation_data=validation_generator,epochs=EPOCHS)

hypermodel = hypermodel
best_hp = tuner.get_best_hyperparameters()[0]
model = hypermodel.build(best_hp)
train_history = hypermodel.fit(hp=best_hp,model=model,refit=True,x=train_generator,validation_data=validation_generator)

save_plot(MODEL_PATH,train_history.history,"loss")

evaluation = model.evaluate(test_generator)
save_pickle(os.path.join(MODEL_PATH,"evaluation.pickle"),evaluation)
predictions = model.predict(test_generator)
original_data = test_generator.transformed_outputs
original_inverted =minmaxscaler.inverse_transform(powertransformer.inverse_transform(original_data))
predicted_inverted =minmaxscaler.inverse_transform(powertransformer.inverse_transform(predictions))

pearsonr = scipy.stats.pearsonr(original_inverted[:,0], predicted_inverted[:,0])
spearman = scipy.stats.spearmanr(original_inverted[:,0], predicted_inverted[:,0])

plot_data(original_inverted, predicted_inverted,os.path.join(MODEL_PATH,MODEL_FOLDER,"first_100.png"))

model.save(os.path.join(MODEL_PATH,"model",MODEL_FOLDER))