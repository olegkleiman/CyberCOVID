import matplotlib.pyplot as plt
import numpy as np

class CityModel:
    def __init__(self, data, regression_model, city_name):
        self.data = data
        self.model = regression_model
        self.name = city_name[::-1]  # reverse this Hebrew string

    def show_regression(self, regressors):
        plt.figure(figsize=(8, 7))
        ax = plt.axes()
        plt.title(self.name)
        ax.scatter(regressors, self.data)

        x_new = np.linspace(0, 30, 100)
        y_new = self.model.predict(x_new[:, np.newaxis])
        ax.plot(x_new, y_new)

        ax.axis('tight')
        plt.show(block=True)