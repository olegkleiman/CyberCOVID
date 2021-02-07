import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


class Model1D:
    def __init__(self, x, y):
        linear_regression_model = linear_model.LinearRegression()
        X = x.reshape(-1, 1)
        Y = y.reshape(-1, 1)

        # split the dataset into training part and test part
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

        linear_regression_model.fit(X_train, Y_train)

        score = linear_regression_model.score(X_test, Y_test)  # check score - the best possible score is 1.0

        self.data = y
        self.model = linear_regression_model
        self.score = score

    def __str__(self):
        return 'Score: {}. Intercept: {} Coefficient: {}'.format(self.score, self.model.intercept_[0],
                                                                 self.model.coef_[0, 0])

    def predict(self, regressors):
        predictions = self.model.predict(regressors)
        return predictions
