import matplotlib.pyplot as plt
import numpy as np
import joblib

from model1d import Model1D


class CityModel(Model1D):
    def __init__(self, city_name, data, regression_model, score):
        super().__init__(data, regression_model, score)
        self.name = city_name[::-1]  # reverse this Hebrew string

    def save(self):
        filename = self.name + 'sav'
        joblib.dump(self.model, filename)

    def load(self):
        filename = self.name + 'sav'
        self.model= joblib.load(filename)

    def show_regression(self, regressors):
        plt.isinteractive(True)
        plt.figure(figsize=(8, 7))
        ax = plt.axes()
        plt.title(self.name)
        ax.scatter(regressors, self.data)

        x_new = np.linspace(0, 30, 100)
        y_new = self.predict(self, x_new[:, np.newaxis])
        ax.plot(x_new, y_new)

        ax.axis('tight')
        # ax.xaxis_date()
        # plt.figure.autofmt_xdate()
        plt.show(block=True)
