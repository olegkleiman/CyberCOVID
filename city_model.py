import matplotlib.pyplot as plt
import numpy as np

from model1d import Model1D


class CityModel(Model1D):
    def __init__(self, city_name, data, regression_model, score):
        Model1D.__init__(self, data, regression_model, score)
        self.name = city_name[::-1]  # reverse this Hebrew string

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
