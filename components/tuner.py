from keras_tuner import HyperModel
from tensorflow.keras import Model, layers, metrics
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from components.classification_metrics import f1_m
from tensorflow import keras


class NLPHyperModel(HyperModel):

    def __init__(self, epochs: int, outp_shape: int, maxlen: int, patience: int):
        """Initialization of Tuner component.

        Args:
            epochs (int): Number of epochs.
            outp_shape (int): Shape of last layer.
            maxlen (int): Maximum lenght of a sequence.
            patience (int): Early stopping patience.
        """
        self.outp_shape = outp_shape
        self.epochs = epochs
        self.maxlen = maxlen
        self.patience = patience

    def build(self, hp):
        """Build and compile model

        Args:
            hp (dict): Hyperparameter set.

        Returns:
            keras.Model: Compiled model.
        """
        if self.outp_shape > 2:
            layer_in, layer_out = self._get_inp_outp_layer(hp=hp)
            model = Model(inputs=layer_in, outputs=layer_out)
            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy",
                         metrics.FalsePositives(),
                         metrics.FalseNegatives(),
                         metrics.Precision(),
                         metrics.Recall(), f1_m],
            )
        else:
            layer_in, layer_out = self._get_inp_outp_layer(hp=hp)
            model = Model(inputs=layer_in, outputs=layer_out)
            model.compile(
                optimizer="adam",
                loss="mse",
                metrics=["cosine_similarity", "mae"],
            )
        model.summary()
        return model

    def _get_inp_outp_layer(self, hp):
        """Generate network structe based on the hyperparameter space.

        Args:
            hp (dict): Hyperparameter set.

        Returns:
            tuple: Input and output layer.
        """
        layer_num_LSTM = hp.Int("LSTM_layer_num", 1, 3)
        layer_num_dense = hp.Int("Dense_layer_num", 1, 3)
        embed_dim = hp.Int("Embeeded_dim", int(
            self.maxlen/10), int(self.maxlen/5))

        layer_in = keras.Input(shape=(None,))

        embedding = layers.Embedding(self.maxlen, embed_dim)(layer_in)

        for i in range(layer_num_LSTM):
            if i == 0:
                if i != layer_num_LSTM-1:
                    layer_LSTM = layers.Bidirectional(layers.LSTM(hp.Int(
                        f"LSTM_units_{i}", min_value=50, max_value=150), return_sequences=True))(embedding)
                else:
                    layer_LSTM = layers.Bidirectional(layers.LSTM(
                        hp.Int(f"LSTM_units_{i}", min_value=50, max_value=150)))(embedding)

            else:
                if i != layer_num_LSTM-1:
                    layer_LSTM = layers.Bidirectional(layers.LSTM(hp.Int(
                        f"LSTM_units_{i}", min_value=50, max_value=150), return_sequences=True))(layer_LSTM)
                else:
                    layer_LSTM = layers.Bidirectional(layers.LSTM(
                        hp.Int(f"LSTM_units_{i}", min_value=50, max_value=150)))(layer_LSTM)

        for i in range(layer_num_dense):
            if i == 0:
                layer_dense = layers.Dense(
                    hp.Int(f"dense_units_{i}", min_value=50, max_value=150), activation=hp.Choice(f"Dense_activations_{i}", ["relu", "sigmoid", "selu"]))(layer_LSTM)
            else:
                layer_dense = layers.Dense(
                    hp.Int(f"dense_units_{i}", min_value=50, max_value=150), activation=hp.Choice(f"Dense_activations_{i}", ["relu", "sigmoid", "selu"]))(layer_dense)
        if self.outp_shape > 2:
            layer_out = layers.Dense(
                self.outp_shape, activation="softmax")(layer_dense)
        else:
            layer_out = layers.Dense(
                self.outp_shape, activation=hp.Choice(f"Output_activations", ["relu", "sigmoid", "linear"]))(layer_dense)

        return layer_in, layer_out

    def fit(self, hp, model, refit=False, *args, **kwargs):
        """Fit model. It can be used during the training also.

        Args:
            hp (dict): Hyperparameters.
            model (keras.Model): Builded keras model.
            refit (bool, optional): Is it a tuning, or training. Defaults to False.

        Returns:
            history-object: History of training.
        """
        if refit:
            earlystopping = EarlyStopping(
                monitor='val_loss', mode='min', verbose=1, patience=self.patience)
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001)

            return model.fit(callbacks=[earlystopping, reduce_lr], epochs=self.epochs, *args, **kwargs)
        else:
            return model.fit(*args, **kwargs)
