import numpy as np
import joblib
import plotly.graph_objs as go

from model1d import Model1D


# Whatever regression model is used, this class abstracts it
# from other functionality: displaying, saving etc.
class CityModel(Model1D):
    def __init__(self, city_name, x, y):
        super().__init__(x, y)
        self.name = city_name[::-1]  # reverse this Hebrew string

    def save(self):
        filename = self.name + '.sav'
        joblib.dump(self.model, filename)
        return self

    def load(self):
        filename = self.name + '.sav'
        self.model = joblib.load(filename)

    # def __rshift__(self, ax):
    #     pass

    def display(self, figure, regressors):
        figure.add_trace(go.Scatter(x=regressors.labels, y=self.data,
                                    mode="markers", name='Cases'))

        x_new = np.linspace(0, len(regressors.labels) - 1, 100)
        y_new = super().predict(x_new[:, np.newaxis])
        values = np.concatenate(y_new, axis=0)
        figure.add_trace(go.Scatter(x=regressors.labels, y=values,
                                    mode='lines', name='Prognosis'))

