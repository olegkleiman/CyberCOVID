import mplcursors
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
        filename = self.name + '.sav'
        joblib.dump(self.model, filename)
        return self

    def load(self):
        filename = self.name + '.sav'
        self.model = joblib.load(filename)

    # def __rshift__(self, ax):
    #     pass

    def display(self, axe, regressors, ticks_number):
        axe.title.set_text(self.name)

        # There are too much date labels for the plot.
        # We'll show only 'ticks_number' labels
        tick_indices = (np.linspace(0, len(regressors.labels) - 1, ticks_number))
        tick_indices = tick_indices.astype(int)
        axe.set_xticks(tick_indices)

        filtered_labels = regressors.labels[tick_indices]

        axe.set_xticklabels(filtered_labels, rotation=40)
        axe.scatter(regressors.indices, self.data, label='Cases')

        x_new = np.linspace(0, len(regressors.labels) - 1, 100)
        y_new = super().predict(x_new[:, np.newaxis])
        lines = axe.plot(x_new, y_new, color='red', label='Prognosis')

        axe.legend(fancybox=True, framealpha=1, loc='lower right')
        mplcursors.cursor(lines)
