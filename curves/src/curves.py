import numpy as np
import matplotlib.pyplot as plt # For ploting
from tqdm import tqdm
import datetime


from src.IOhelper import IOhelper
from src.generate import Generate
from src.touchpoints import TouchPoints

"""
Returns a normalized hyperbolic curve. Noise can be added to each curve. 
"""


class Curve:

    def __init__(self, hyp_func=np.sinh, fs=(5, 15), f=2.0, factor=1.0,
                 date_range=(1, 30)):
        self.hyp_func = hyp_func
        self.fs = np.random.randint(fs[0], fs[1])
        self.f = f
        self.factor = factor
        self.date_range = date_range
        self.x = None
        self.y = None
        self.days = None

    def hyperbolic(self):
        """
        Creates a hyperbolic wave where function, sample rate and frequency decides the structure of the curve.
        :return:
        """
        # the points on the x axis for plotting

        self.x = np.arange(self.fs)
        # compute the value (amplitude) of the sin wave at the for each sample
        self.y = self.factor*self.hyp_func(2*np.pi*self.f * (self.x/self.fs))
        self._normalize()
        self.time_distribution()


    def plot_curve(self):
        plt.stem(self.x, self.y, 'r')
        plt.plot(self.x, self.y)
        plt.title("name: " + self.hyp_func.__name__ + ",rate: " + str(self.fs) + ",frequency: " + str(self.f)
                  + ",range_mult: " + str(self.factor))
        plt.show()

    def add_noise(self, target_snr=15):
        """
        AWGN Noise based on a desired SNR ( signal to noise ratio )
        :param target_snr: The target signal ratio, for example: higher number means less noise.
        :return:
        """
        distribution = np.mean(self.y)
        sig_avg = 10 * np.log10(distribution)

        noise_avg_db = sig_avg - target_snr
        noise_avg_= 10 ** (noise_avg_db / 10)

        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_avg_), len(self.x))
        # Noise up the original signal
        self.y = self.y + noise
        self._normalize()

    def _normalize(self):
        self.y = (self.y - self.y.min(0)) / self.y.ptp(0)

    def time_distribution(self):
        """
        Initialize a timestep for each entry in the given curve. then sorts it to simulate a
        user paths in time. Rewrite
        :param range:tuple(int, int) or (float, float)
                     - The range ffor the timesteps
        :param curve: List ( int )
                     - Each entry correspondent to one touch point (event)
        :return:
        """
        # Add zero first, declaring start. Then add the rest of the random distribution.
        temp1 = np.array([0])

        # Fill with random values in range
        temp2 = np.array(self.date_range[0] + np.random.sample(len(self.x) - 1)
                         * self.date_range[1], dtype='int')

        self.days = np.concatenate((temp1, temp2))
        # Convert to unix timestamps
        self._days()

        # Sort in ascending order
        self.days.sort()

    def _days(self):
        """
        Converts a day in to unix timestamp,
        :return:
        """
        oneday_unix = 86400
        for index, day in enumerate(self.days):
            self.days[index] = day * oneday_unix


def create_curves(f_list=[0.5, 1.0], factor_list=[0.5, 1.0], fs=(5,10),
                  hyp_funcs=[np.sinh, np.sin, np.cosh, np.cos, np.tanh]):
    """
    Loops trough each setting and creates a hyp function for each setting.
    :param f_list:
    :param factor_list:
    :param fs:
    :param hyp_funcs:
    :return:
    """

    curves = []

    for func in hyp_funcs:
        for f in f_list:
            for factor in factor_list:
                curve = Curve(func, fs, f, factor)
                curves.append(curve)
    return curves


def main():

    f_list = [0.5, 1.0]
    factor_list = [0.5, 1.0]
    fs = (5, 15)
    hyp_func_list = [np.sinh, np.sin, np.cosh, np.cos, np.tanh]

    curves = create_curves(f_list, factor_list, fs, hyp_func_list )

    touch_dict = {
        "social": 1,
        "direct": 2,
        "organic": 3,
        "display:": 4,
        "video": 5,
        "email": 6,
        "other": 7
    }

    touchpoints = TouchPoints(curves=curves, touchpoint_names=touch_dict, conversion_rate=0.4)
    curves[10].hyperbolic()
    curves[10].plot_curve()
    curves[10].add_noise(15)
    curves[10].plot_curve()


    distributions = []
    answers = []
    nr_samples = 100000

    gen = Generate(touchpoints, curves, snr_interval=(16, 20))
    paths, times, answers = gen.generate_distribution(100000)

    print(paths[0])
    print(len(paths[0]))
    print(paths[1])
    print(times[0])
    print(times[1])
    print(answers[0])
    print(answers[1])
    print("len of paths:", len(paths[0]), " - len of times", len(times[0]))



    helper = IOhelper("../data/")
    helper.write_2d_csv(paths, "paths")
    helper.write_2d_csv(times, "times")
    helper.write_1d_csv(answers, "answers")
    helper.write_dict_to_csv(touchpoints.names, "vocabulary")
    helper.dump_json(touch_dict, "vocabulary")



if __name__ == "__main__":
    main()


# TODO: Next Phase
#  1. Create time data for each step,
#  2. Make the paths random, which means that they can be of length [n1, n2]
#  3- Create Three different characeristics?
#  4. Implement the CIU Module.

