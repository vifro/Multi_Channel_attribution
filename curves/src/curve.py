import numpy as np
import matplotlib.pyplot as plt # For ploting




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
        self.shifter = (-3, 3)

    def hyperbolic(self):
        """
        Creates a hyperbolic wave where function, sample rate and frequency decides the structure of the curve.
        :return:
        """
        # the points on the x axis for plotting

        self.x = np.arange(self.fs)
        # compute the value (amplitude) of the sin wave at the for each sample
        self.y = self.factor*self.hyp_func(2*np.pi*self.f * (self.x/self.fs)) + np.random.randint(self.shifter[0], self.shifter[1])
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

        # Sort in descending  order
        self.days = -np.sort(-self.days)

    def _days(self):
        """
        Converts a day in to unix timestamp,
        :return:
        """
        oneday_unix = 86400
        for index, day in enumerate(self.days):
            self.days[index] = day * oneday_unix




