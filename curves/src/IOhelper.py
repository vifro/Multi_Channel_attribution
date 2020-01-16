import csv
import os.path
from os import path
import json


class IOhelper:
    """
    Simple class for reading and writing 2d arrays in to a .csv
    """
    def __init__(self, path_name, google_colab=False):
        if not google_colab:
            if self.check_path(path_name):
                self.path_name = path_name
            else:
                raise IOError("Not a valid directory")
        else:
            self.path_name = path_name

    def write_2d_csv(self, data, name):
        with open(self.path_name + name + ".csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def write_1d_csv(self, data, name):
        with open(self.path_name + name + ".csv", "w", newline="") as f:
            writer = csv.writer(f)
            for list_ in data:
                writer.writerow((str(int(list_))))

    def write_dict_to_csv(self, data, name):
        with open(self.path_name + name + ".csv", 'w', newline="") as csv_file:
            writer = csv.writer(csv_file)
            for key, value in data.items():
                writer.writerow([key, value])

    def read_csv(self, name):
        return list(csv.reader(open(self.path + name)))

    @staticmethod
    def check_path(path_name):
        return path.isdir(path_name)

    def dump_json(self, sample, name):
        with open(name + '.json', 'w') as fp:
            json.dump(sample, fp)