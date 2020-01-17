from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, LSTM, RepeatVector, Concatenate, Activation, Dot, Subtract, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K


class AttributionModel:

    def __init__(self, config, data):
        """

        :param config: Dictionary containing configurations:
            - ['vocab_size']    Vocabulary size
            - ['seq_len']       Maximum sequence length
            - ['hidden_len']    Hidden length of the time layer
            - ['lstm_units']    Number of LSTM Units
            - ['dense_units']   Number of hidden units in the ouput layer
            - ['learning_rate'] Learning rate
            - ['batch_size']    Batch size for training
            - ['epochs']        Epochs for training
            - ['class_weights'] Class weights to desribe the ratio of classes.

        :param data: Dictionary containing the data.
            - ['X_train']       User paths
            - ['X_val']
            - ['X_test']
            - ['X_train_time']  The corresponding times for each training path
            - ['X_val_time']
            - ['X_test_time']
            - ['y_train']       The answers
            - ['y_val']
            - ['y_test']
        """
        # Parameters
        self.vocab_size = config['vocab_size']
        self.seq_len = config['seq_len']
        self.hidden_len = config['hidden_len']
        self.lstm_units = config['lstm_units']
        self.dense_units = config['dense_units']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.class_weights = config['class_weights']

        # Data
        self.X_train = data['X_train']
        self.X_val = data['X_val']
        self.X_test = data['X_test']
        self.X_train_time = data['X_train_time']
        self.X_val_time = data['X_val_time']
        self.X_test_time = data['X_test_time']
        self.y_train = data['y_train']
        self.y_val = data['y_val']
        self.y_test = data['y_test']

        # Initialized matrix of np.zeroes(training_samples, lstm_units)
        print(self.X_train.shape[0])
        print(self.lstm_units)
        self.s0 = np.zeros((self.X_train.shape[0], self.hidden_len))
        self.s1 = np.zeros((self.X_val.shape[0], self.hidden_len))
        self.s2 = np.zeros((self.X_test.shape[0], self.hidden_len))

        self.model = None
        self.history = None

    def build_model(self):
        """
        NOTE: build LSTM model
            Inputs:
            max_features: number of features
            X_train_pad: input data after padding
            y_train: label of training data
            X_test_pad: testing data after padding
            y_test: label of testing data
            Outputs:
            eval_score: evaluated score
            eval_acc: evaluated accuracy
            model: lstm model
        """
        p = Input(shape=(self.seq_len, self.vocab_size), name="input_path")  # Seq_len
        s = Input(shape=(self.hidden_len,), name="hidden_state")             # n_s
        t = Input(shape=(self.seq_len, 1), name="input_time")

        a = LSTM(self.lstm_units, return_sequences=True)(p)                 #20, 64

        context = self.time_attention(a, s, t)
        c = Flatten()(context)
        out_att = Dense(self.dense_units, activation="sigmoid", name="path_outcome")(c)
        output = Dense(1, activation="sigmoid", name="binary_output")(out_att)

        self.model = Model([p, s, t], output)
        print(self.model.summary())

    def softmax(self, x, axis=1):
        ndim = K.ndim(x)
        if ndim == 2:
            x = K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            x = e / s
        else:
            raise ValueError("1D tensor not working on softmax")

        return x

    def train_model(self, loss="binary_crossentropy", opt=Adam, metrics="accuracy"):
        self.model.compile(loss=loss,
                           optimizer=opt(lr=self.learning_rate),
                           metrics=[metrics]
                           )
        self.history = self.model.fit(x=[self.X_train,  self.s0, self.X_train_time],
                                      y=self.y_train,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      verbose=2,
                                      validation_data=([self.X_val, self.s1, self.X_val_time], self.y_val)
                                      )

    def evaluate_model(self):
        """
        Evaluates the score and prints the metrics and the corresponding score
        @return:
        """
        scores = self.model.evaluate([self.X_test, self.s2, self.X_test_time], self.y_test, verbose=1)
        print("%s: %.2%%" % (self.model.metrics_names[1], scores[1]*100))

    def save_model(self, name="attribution_model.h5"):
        self.model.save_weights(name)

    def load_model(self, name="attribution_model.h5"):
        self.model.load_weights(name)

    def time_attention(self, a, s_prev, t0):
        """
        The time attention layer is described in the paper. But short the equation
        calculating the output layer becomes
                     exp(vt*u - T_t)
            a_t = ---------------------     (1.1)
                   sum(exp(vt*u - T_t)
        where a small layer is added before to compute the concatenated values from
            * a: output from hidden layer
            * s_prev: Hidden state
        @param a: Output from previous lstm sequence (hidden state
        @param s_prev: The hidden state
        @param t0: The times corresponding the the user path.
        @return: A vector of tensors
        """
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a".
        # Repeat vector so that s_prev can be concatenated
        s_prev = RepeatVector(self.seq_len)(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis
        concat = Concatenate(axis=-1)([s_prev, a])
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e.
        e = Dense(10, activation="tanh")(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies.
        energies = Dense(1, activation="relu")(e)
        # Use "activator" on "energies" to compute the attention weights "alphas"
        energies = Subtract()([energies, t0])
        activator = Activation(self.softmax, name='attention_weights')
        alphas = activator(energies)
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next layer
        return Dot(axes=1)([alphas, a])

    def CIU(self):
        pass

    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),
                         num_classes=num_classes)


def main():
    """
    Just and example on how to run the model.
    :return:
    """
    seq_len = 2
    hidden_len = 32
    lstm_units = 64
    dense_units = 32
    batch_size = 64
    learning_rate = 0.001
    epochs = 40

    channels = ['Organic Search', 'Paid Search', 'Social', 'display', 'Video', 'Social', 'Other']
    nr_channels = len(channels)
    vocab_size = nr_channels + 1

    config = {
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "hidden_len": hidden_len,
        "lstm_units": lstm_units,
        "dense_units": dense_units,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "class_weights": {0: 0.5, 1: 1.5},
    }
    # Binary encoder
    X = [[0, 1], [2, 3], [4, 5], [6, 7]]
    Xt = np.array([[200, 300], [0, 100], [14, 1200], [230, 400]])
    y = np.array([0, 1, 0, 1])
    X = np.array(list(map(lambda x: to_categorical(x, num_classes=vocab_size), X)), ndmin=3)
    print(Xt.shape)
    Xt = Xt.reshape(-1, seq_len, 1)

    print(X.shape)
    print(Xt.shape)
    data = {
        "X_train": X,
        "X_val": X,
        "X_test": X,
        "X_train_time": Xt,
        "X_val_time": Xt,
        "X_test_time": Xt,
        "y_train": y,
        "y_val": y,
        "y_test": y
    }

    attrmod = AttributionModel(config=config, data=data)
    attrmod.build_model()
    attrmod.train_model()


if __name__ == "__main__":
    main()