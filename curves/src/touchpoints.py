import numpy as np


class TouchPoints:

    def __init__(self, curves,  touchpoint_names, conversion_rate=0.4, time=24):
        """
        Creates information for generating touch points.
        :param curves: List,
                        containing hyperbolic functions
        :param touchpoint_names: list
                                    Containing touchpoint names
        :param conversion_rate:
        :param time: integer,
                    Upper bound for hours/minutes or whatever
        """
        self.time = time
        self.curves = curves
        self.names = touchpoint_names
        self.conversions = self.conversion(conversion_rate=conversion_rate)
        self.values = self.touchpoint_values()
        self.translator = self.reverse()

    def reverse(self):
        """
        Reverse lists of key and values
        :return:
        :rtype:
        """
        keys = list(self.values.values())
        values = list(self.names.values())
        return dict(zip(keys, values))

    def touchpoint_values(self):
        """
        Get touchpoint values
        :return: Touchpoint values evenly divided between (0,1)
        """
        touchpoint_values = {}
        nr_touchpoints = len(self.names)
        for index, key in enumerate(self.names.keys()):
            touchpoint_values[key] = ((index+1)/nr_touchpoints)
        return touchpoint_values

    def conversion(self, conversion_rate):
        """
        Randomize conversions
        :param conversion_rate:
        :return: List of conversions
        """
        conversions = []
        nums = np.zeros(len(self.curves), dtype=bool)
        last = np.math.ceil(len(self.curves) * conversion_rate)

        # Convert to True
        nums[:last] = True

        # Shuffle list
        np.random.shuffle(nums)

        # Fill list
        for index in range(len(self.curves)):
            conversions.append(nums[index])
        return conversions

    def random_curve(self):
        """
        Random Curve
        :return:
        """
        index = np.random.randint(0, len(self.curves)-1)

        return self.curves[index], self.conversions[index]
