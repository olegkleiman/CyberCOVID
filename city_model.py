import matplotlib.pyplot as plt
import numpy as np
import joblib

from model1d import Model1D


# Whatever regression model is used, this class abstracts it
# from other functionality: displaying, saving etc.
class CityModel(Model1D):
    def __init__(self, city_name, data, regression_model, score):
        super().__init__(data, regression_model, score)
        self.name = city_name[::-1]  # reverse this Hebrew string

    def save(self):
        filename = self.name + 'sav'
        joblib.dump(self.model, filename)

    def load(self):
        filename = self.name + 'sav'
        self.model = joblib.load(filename)

    # def __rshift__(self, ax):
    #     pass

    def display(self, ax, regressors):
        ax.title.set_text(self.name)
        ax.scatter(regressors, self.data)

        x_new = np.linspace(0, 30, 100)
        y_new = super().predict(x_new[:, np.newaxis])
        ax.plot(x_new, y_new, color='red')
        ax.set_title(self.name)

