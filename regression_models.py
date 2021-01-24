from city_model import CityModel
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import tensorflow as tf

# TensorFlow implementation  of the Linear Regression
class SimpleLinearRegression:
    def __init__(self, initializer='random'):
        if initializer == 'ones':
            self.var = 1.
        elif initializer == 'zeros':
            self.var = 0.
        elif initializer == 'random':
            self.var = tf.random.uniform(shape=[], minval=0., maxval=1.)

        self.m = tf.Variable(1., shape=tf.TensorShape(None))
        self.b = tf.Variable(self.var)

    def predict(self, x):
        return tf.reduce_sum(self.m * x, 1) + self.b

    def mse(self, true, predicted):
        return tf.reduce_mean(tf.square(true - predicted))

    def update(self, X, y, learning_rate):
        with tf.GradientTape(persistent=True) as g:
            loss = self.mse(y, self.predict(X))

        print("Loss: ", loss)

        dy_dm = g.gradient(loss, self.m)
        dy_db = g.gradient(loss, self.b)

        self.m.assign_sub(learning_rate * dy_dm)
        self.b.assign_sub(learning_rate * dy_db)

    def train(self, x, y, learning_rate=0.01, epochs=5):
            X = x.reshape(-1, 1)
            Y = y.reshape(-1, 1)

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

            if len(X.shape) == 1:
                X = tf.reshape(X, [X.shape[0], 1])

            self.m.assign([self.var] * X.shape[-1])

            for i in range(epochs):
                print("Epoch: ", i)

                self.update(X, y, learning_rate)

# SkLearn implementation of the Linear Regression

def calculate_regression_params(x, y, name):
    model = linear_model.LinearRegression()
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)

    # split the dataset into training part and test part
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)  # check score - the best possible score is 1.0
    model = CityModel(city_name=name, data=y, regression_model=model, score=score)
    print(model)  # __str__ if redefined in Model1D

    return model


# TODO: implement it for using with apply_along_axis()
def calc(row):
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    _model = linear_model.LinearRegression()

    x = dates.indices
    X = x.reshape(-1, 1)
    y = row[2:]
    Y = y.reshape(-1, 1)

    # split the dataset into training part and test part
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
    _model.fit(x_train, y_train)
    score = _model.score(x_test, y_test)
    print('Score: {}. Intercept: {} Coefficient: {}'.format(score, _model.intercept_[0], _model.coef_[0, 0]))

    return CityModel(data=y, regression_model=model, city_name=row[0], score=score)
