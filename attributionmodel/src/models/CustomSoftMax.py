from keras.engine import Layer
from keras import backend as K


class CustomSoftMax(Layer):
    """
    Custom Softmax
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CustomSoftMax, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(CustomSoftMax, self).build(input_shape)  # Be sure to call this at the end

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, axis=1):
        ndim = K.ndim(x)
        if ndim == 2:
            x = K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            x = e / s
        else:
          raise ValueError("1D tensor not working on softmax")

        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)