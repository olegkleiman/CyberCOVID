import numpy as np


class Model1D:
    def __init__(self, data, regression_model, score):
        self.data = data
        self.model = regression_model
        self.score = score

    def __str__(self):
        return 'Score: {}. Intercept: {} Coefficient: {}'.format(self.score, self.model.intercept_[0], self.model.coef_[0, 0])

    def predict(self, regressors):
        predictions = self.model.predict(regressors)
        return predictions