import csv
import random
import numpy as np


class Data:
    def __init__(self, files):
        self.data = []
        for file in files:
            print("Reading {}".format(file))
            self.data += self.read(file)
        self.days = int(len(self.data) / 240)

    def read(self, file):
        result = []
        with open(file, newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            for _, row in enumerate(spamreader):
                if _:
                    result.append([float(i) for i in row[1:]])
        return result

    def gen_batch_by_day(self, start_day):
        x = np.zeros([8 * 240, 6])
        for i in range(0, 8 * 240):
            x[i] = self.data[start_day * 240 + i]
        max = np.max(x, axis=0)
        x_ = x.transpose().copy()
        for i in range(len(max)):
            x_[i] = x_[i] / max[i] * 0.6
        x_ = x_.transpose()
        return x_, x

    def gen_one_batch(self):
        def FOLDtoONEHOT(fold):
            fold = fold - 1
            distribution = [i / 100 for i in range(-10, 11)]
            for i, _ in enumerate(distribution[:-1]):
                if distribution[i] <= fold < distribution[1 + i]:
                    return np.array([1 if j == i else 0 for j in range(len(distribution) - 1)])
            if fold >= 1.1:
                return np.array([1 if j == 19 else 0 for j in range(len(distribution) - 1)])
            if fold <= 0.9:
                return np.array([1 if j == 0 else 0 for j in range(len(distribution) - 1)])

        x = np.zeros([7 * 240, 6])
        y = np.zeros([1 * 240, 1])
        start_day = random.randint(0, self.days - 8)
        for i in range(0, 7 * 240):
            x[i] = self.data[start_day * 240 + i]
        for i in range(0, 1 * 240):
            y[i] = self.data[(start_day + 7) * 240 + i][2]
        max = np.max(x, axis=0)
        x_ = x.transpose().copy()
        for i in range(len(max)):
            x_[i] = x_[i] / max[i] * 0.6
        x_ = x_.transpose()
        fold = np.max(y) / x[-1][0]
        # y_ = np.array([1 if 1 + i / 10 < fold < 1 + (i + 1) / 10 else 0 for i in range(-5, 5)])
        y_ = FOLDtoONEHOT(fold)
        return x_, y_, x, y

    def next_batch(self, amount=100):
        x = np.zeros([amount, 7 * 240, 6])
        y = np.zeros([amount, 20])
        for i in range(amount):
            x[i], y[i], _, _ = self.gen_one_batch()
        return x, y


if __name__ == "__main__":
    dataset = Data(["data/data_{}-1-1_{}-1-1.csv".format(i, i + 1) for i in range(2009, 2016)])
    x, y = dataset.next_batch()
    # for i in y:
    #     print(list(i).index(1))
    # print(x.transpose().shape)
    # print(x.shape, y.shape)
