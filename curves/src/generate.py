import numpy as np
from tqdm import tqdm


class Generate:

    def __init__(self, touchpoint, curves, snr_interval=(13, 17)):
        self.touchpoint = touchpoint
        self.curves = curves
        self.snr_interval = snr_interval

    def hyperbolic_curve(self):
        """
        Get a random hyperbolic function from touchpoints, create the curve with
        the settings in touchpoint and add noise.
        :return: The curve points, the number of days(in specified time accruacy)
                 and if it was a conversion or not.
        :rtype: tuple(list of integers, list of integers, boolean)
        """
        curve, conversion = self.touchpoint.random_curve()
        curve.hyperbolic()
        curve.add_noise(np.random.randint(self.snr_interval[0], self.snr_interval[1]))
        return curve.y, curve.days, conversion

    def convert_to_touchpoints(self, hyp_curve):
        """
        Conver the generate curve to specified touchpoints.
        :param hyp_curve: The curve to be converted.
        :return:
        """
        y = np.zeros(len(hyp_curve))
        touch_values = list(self.touchpoint.values.values())
        for index, value in enumerate(hyp_curve):
            y[index] = touch_values[np.abs(touch_values-value).argmin()]
        return self._translate(y)

    def _translate(self, to_translate):
        """
        Translate the values in the curve to touchpoints
        :param to_translate:
        :return:
        """
        return [self.touchpoint.translator[x] for x in to_translate]

    def generate_distribution(self, nr_samples):
        """
        Generate the distribution.
        :param nr_samples: Number of samples
        :return: Touchpoints( each interger represents a channel
                 Times ( unix Timestamp)
                 Conversion ( boolean
        """
        touchpoints = []
        answers = []
        times = []
        for i in tqdm(range(nr_samples)):
            hyp_curve, days, conversion = self.hyperbolic_curve()
            touchpoints.append(self.convert_to_touchpoints(hyp_curve))
            times.append(days)
            answers.append(conversion)

        return touchpoints, times, answers


