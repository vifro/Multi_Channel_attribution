from src.IOhelper import IOhelper

import numpy as np

from src.curve import Curve
from src.generate import Generate
from src.touchpoints import TouchPoints


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

