class Model1D:
    def __init__(self, data, regression_model, score):
        self.data = data
        self.model = regression_model
        self.score = score

    def predict(self, regressors):
        predictions = self.model.predict(regressors[:, np.newaxis])
        return predictions