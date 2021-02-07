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
        figure.add_trace(go.Line(x=regressors.labels, y=values, name='Prognosis'))

        # axe.title.set_text(self.name)
        # axe.tick_params(axis='both', which='major', labelsize=8)
        #
        # # There are too much date labels for the plot.
        # # We'll show only 'ticks_number' labels
        # tick_indices = np.linspace(0, len(regressors.labels) - 1, ticks_number)
        # tick_indices = tick_indices.astype(int)
        # axe.set_xticks(tick_indices)
        #
        # filtered_labels = regressors.labels[tick_indices]
        #
        # axe.set_xticklabels(filtered_labels, rotation=40)
        # axe.scatter(regressors.indices, self.data, label='Cases')
        #
        # x_new = np.linspace(0, len(regressors.labels) - 1, 100)
        # y_new = super().predict(x_new[:, np.newaxis])
        # lines = axe.plot(x_new, y_new, color='red', label='Prognosis')
        #
        # axe.legend(fancybox=True, framealpha=1, loc='lower right')
        # mplcursors.cursor(lines)
