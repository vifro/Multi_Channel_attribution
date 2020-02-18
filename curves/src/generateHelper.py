import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class GenerateHelper():

    def __init__(self, seqlen, vocab_size):
        self.seqlen = seqlen
        self.vocab_size = vocab_size

    def data_set_generator(self, paths, times, vocab_size):
        all_path_steps = []
        all_time_steps = []
        for i in range(self.seqlen):
            p, t = self.get_timestep(paths, times, i, vocab_size)
            all_path_steps.append(p)
            all_time_steps.append(t)
        return np.array(all_path_steps), np.array(all_time_steps)

    @staticmethod
    def create_data_config(all_x_train, all_xt_train, y_train,
                           all_x_val, all_xt_val, y_val,
                           all_x_test, all_xt_test, y_test):
        all_data = []
        for i, value in enumerate(all_x_train):
            data = {
                "X_train": all_x_train[i],
                "X_val": all_x_val[i],
                "X_test": all_x_test[i],
                "X_train_time": all_xt_train[i],
                "X_val_time": all_xt_val[i],
                "X_test_time": all_xt_test[i],
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test
            }
            all_data.append(data)
        return all_data

    @staticmethod
    def get_timestep(paths, times, timestep, vocab_size):
        """
        Write something
        """
        if len(paths) < timestep or timestep < 0:
            print("Not a valid timestep value")
            return -1
        else:
            seqlen = len(paths[0])
            feature_p = []
            feature_t = []
            for index, x in enumerate(paths):
                path = np.array([[0.] * vocab_size] * seqlen)
                time = np.array([0.] * seqlen)
                # Just add the ones that
                if not np.array_equal(x[timestep], path):
                    path[timestep] = x[timestep]
                    time[timestep] = times[index][timestep]
                    feature_p.append(path)
                    feature_t.append(time)
            # Convert to numpy and reshape
            feature_p = np.array(feature_p)
            feature_t = np.array(feature_t)
            feature_t = feature_t.reshape(-1, len(feature_p[0]), 1)

            return feature_p, feature_t

    def print_info(self, feature_len, total_len, feature):
        print("Occurrences -", feature, ":", feature_len,
              "(", int(round(feature_len / total_len, 2) * 100), "%)")

    def create_feature_paths(self, indexes, paths, times, feature):
        """
        Creates a time and path for each fucking user.

        INPUT:
        :param indexes:
        :param paths:
        :param times:
        :param features:
        RETURN:
        :paths_feature: all paths, for one specific feature.
        :times_feature: all times for one specific feature.
        """
        temp = []
        feature_p = []
        feature_t = []
        for user_path in indexes[feature - 1]:
            if user_path[1]:
                feature_p.append([paths[user_path[0]][x] for x in user_path[1]])
                feature_t.append([times[user_path[0]][x] for x in user_path[1]])
        self.print_info(len(feature_p), len(indexes[feature - 1]), feature)
        return feature_p, feature_t

    @staticmethod
    def split(data, answers, validation_size=0.2, test_size=0.2):
        """

        """
        # Split training in to training, validation and test sets
        X_train, X_test, y_train, y_test = train_test_split(data, answers,
                                                            test_size=test_size,
                                                            random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                          y_train,
                                                          test_size=validation_size,
                                                          random_state=1)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_time(self, time_type, days):
        """
        Generate a timely distribution over a certain amount of days,
        choose between, hours, minutes or seconds evenly distributed over
        the days.
        :param time_type: (int) 1 for hours, 2 ,for minutes, 3 for seconds.
        :param days: how many days
        """
        seconds = 60
        minutes = 60
        hours = 24

        if time_type is 1:
            time_dist = np.array([[i] for i in range(hours*days + 1)])*minutes*seconds
        elif time_type is 2:
            time_dist = np.array([[i] for i in range(hours*days*minutes + 1)])*seconds
        elif time_type is 3:
            time_dist = np.array([[i] for i in range(hours*days*minutes*seconds + 1)])
        else:
            print("Not a valid time_type")
            raise Exception("Time type exception")
        return time_dist

    def mock_data(self, timestep, features, time_type, nr_days):
        """
        Creates each possible outcome for one specific timestep

        :param timestep: in which step of time the values should be generated
        :param features: The number of possible features
        :param time_type:(int) 1 for hours, 2 ,for minutes, 3 for seconds.
        :param nr_days: ( int)
        """
        paths = []
        times = []

        for feature in features:
            for i in self.get_time(time_type, timestep[1]):
                # Create the times series
                path_series = np.zeros(timestep[1])
                time_series = np.zeros(timestep[1])
                # Add the values
                path_series[timestep[0]] = feature
                time_series[timestep[0]] = i
                paths.append(path_series)
                times.append(time_series)
        times = np.array(times)
        times = times.reshape(-1, len(paths[0]), 1)
        paths = np.array(
            list(map(lambda x: to_categorical(x, num_classes=self.vocab_size),
                     paths)), ndmin=3)
        return paths, times
