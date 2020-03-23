import numpy as np


class CIU:

    def __init__(self):
        pass

    def calculate_CU(self, y, dymin, dymax):

        """
                 y - cmin_x(c_i)
        CU = -------------------------
               cmax_x(c_i) - cmin_x(c_i)
        Where,
            c_i:        is the context studied( which defines the fixed input
                        values of the model.
            x:          is the input(s) for which CU are calculated, so it may
                        also be a vector
            y:          is the output value for the output j studied when the
                        inputs are those defined by c_i
            c_max(c_i): is highest output values observed by varying the value
                        of the inputs.
            c_min(c_i): is lowest output values observed by varying the value
                        of the inputs

        :param y:
                     -Is the output value for the output j studied when the inputs
                       are those defined by C_i
        :param dymin: c_min(c_i)
                     - Lowest output values observed by varying the value for
                       the inputs x
        :param dymax: c_max(c_i)
                     - Highest output values observed by varying the value of the
                       inputs x
        :return: The Contextual Utility
        """
        cu = np.divide((y - dymin), (dymax - dymin))
        return cu

    def calculate_CI(self, dynmax, dynmin, absmax, absmin):
        """
              dynmax(c_i) - dynmin(c_i)
        CI = ---------------------------
                 absmax - absmin
        Where,
            dynmax: is the highest

        :param dynmax:
        :param dynmix:
        :param absmin:
        :param absmax:
        :return:
        """
        ci = np.divide((dynmin - dynmax), (absmax - absmin))
        return ci
