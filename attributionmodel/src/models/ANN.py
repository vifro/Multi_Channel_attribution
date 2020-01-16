from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


INPUT, HIDDEN, DROPOUT, OUTPUT = 1, 2, 3, 4

class ANN():

    def __init__(self, input_size, layers, output_layer, dropout):
        self.input_size = input_size
        self.layers = layers
        self.output_layer = output_layer
        self.dropout = dropout

    def nn_model(self):
        """
        This model builds a model with specified parameters.

        @param input_size:
        @param nr_features:
        @param layers: Example, [(3, "sigmoid"), (2, "sigmoid")]. First is nr of neourons and second is the activation
                       function
        @param output_layer: (nr_features, how many layers)
        @param dropout: list of tuples. same size as layers. Describes if dropout should be added or not.
                        (0, 0) means no dropout, (1, 0.2) means dropout with 0.2 after the Dense layer on same index
                        in layers list.
        @return: A model to be compiled / Concatenated.
        """
        if not len(self.layers) == len(self.dropout):
            raise Exception("Dropout and layers does not have the same length")

        inputs = Input(shape=(self.input_size,), name=self.create_name(INPUT))
        first = True
        holder = None

        for index, layer in enumerate(self.layers):
            if first:
                first = False
                holder = Dense(layer[0], activation=layer[1], name=self.create_name(HIDDEN, index))(inputs)
                if not self.dropout[index][0] == 0:
                    holder = Dropout(self.dropout[index][1], name=self.create_name(DROPOUT, self.dropout[index][1]))(holder)
            else:
                holder = Dense(layer[0], activation=layer[1], name=self.create_name(HIDDEN, index))(holder)
                if not self.dropout[index][0] == 0:
                    holder = Dropout(self.dropout[index][1], name=self.create_name(DROPOUT, self.dropout[index][1]))(holder)

        output = Dense(self.output_layer[0], activation=self.output_layer[1], name=self.create_name(OUTPUT))(holder)

        return Model(inputs=inputs, outputs=output)

    def create_name(self, layer_type, nr=1):
        """
        Creates a name for the layer. Can either be input, hidden_nr or output
        @param layer_type:
        @param nr:
        @return:
        """
        if layer_type == 1:
            return "Input"
        elif layer_type == 2:
            return "Hidden_" + str(nr)
        elif layer_type == 3:
            return "Dropout_" + str(nr)
        elif layer_type == 4:
            return "Output"
        else:
            raise Exception("Not the correct type for creating a name")