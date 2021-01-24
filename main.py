# CyberCOVID project.

# Project aims: TODO

import io
import numpy as np
import pandas as pd
import requests
import joblib

# Flask stuff
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from city_model import CityModel
from enumerated_dates import EnumeratedDates
import matplotlib.pyplot as plt

print('{} version: {}'.format(np.__name__, np.__version__))
print('{} version: {}'.format(pd.__name__, pd.__version__))

#
# 1. Download the raw data into Pandas DataFrame.
#   Note that is raw data in csv format and it differs from the data as displayed
#   by GitHub at https://github.com/idandrd/israel-covid19-data/blob/master/CityData.csv

url = "https://raw.githubusercontent.com/idandrd/israel-covid19-data/master/CityData.csv"
s = requests.get(url).content

df = pd.read_csv(io.StringIO(s.decode('utf-8')))

#
# 2. Prepare the downloaded data for further analysis
#
# We're going to calculate the regression params for each
# city (row) in the dataset. The input for the regression is:
#   - the set (numpy array) of the infected cases for each date
#   - the dates as defined in the DataFrame's first row
#   * Note that the dates were downloaded in "dates" format
#       and in order to serve as such the input, the dates values should be
#       converted to numbers (see Step 2b)

# 2a. Throw out nan values and other mess from the downloaded DataFrame.
#   Thanks to God, Pandas has built-in functions for such a purpose
# Output of Step 2a: df
df.fillna(0, inplace=True)
df.replace('-', 0., inplace=True)  # prefer mutable version because it modifies other views
# print(df)

#
# 2b. Prepare the dates obtained from the DataFrame to participate in regression:
#   just enumerate all the dates, i.e. convert them to the running number
# Output of Step 2b: enumeratedDates array

keys = df.keys()
dates = EnumeratedDates(keys[2:])


#
# 3. Calculate the regression model parameters.
# The function below will be called for each city (row) in the prepared dataset
#
def calculate_regression_params(x, y, name):
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    model = linear_model.LinearRegression()
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)

    # split the dataset into training part and test part
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)  # check score - the best possible score is 1.0
    print('Score: {}. Intercept: {} Coefficient: {}'.format(score, model.intercept_[0], model.coef_[0, 0]))

    # filename = 'finalized_model.sav'
    # joblib.dump(model, filename)

    return CityModel(y, model, name, score)


# Our Linear Regression model is based on SkLearn implementation which accept numpy arrays
# Hence, we convert the prepared DataFrame into numpy array (and intentionally losing the captions raw)
ndata = df.to_numpy()
models = np.array([])

for row in ndata:
    # First two columns are defined as 'City' and 'Population'. We skip them
    model = calculate_regression_params(dates.indices, row[2:], row[0])
    models = np.append(models, model)

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

_models = np.apply_along_axis(calc, 1, ndata)

# Just show some regressions for largest cities
# if the model's score is acceptable according to score
threshold = 0.9
with dates:
    # TODO
    # plt.figure(figsize=(8, 7))
    # fig, axs = plt.subplots(4)
    # fig.suptitle('Infected spreads in most populated cities')
    # x_new = np.linspace(0, 30, 100)
    # for i in np.arange(0, 4):
    #     y_new = models[i].predict(dates.labels)
    #     axs[i].plot(x_new, y_new)
    #     axs[i].set_title(models[i].name)
    # plt.show()

    for i in np.arange(0, 4):
        if _models[i].score > threshold:
            _models[i].show_regression(dates.labels)

# STYLE = [dbc.themes.FLATLY]
# app = dash.Dash('Cyber COVID', external_stylesheets=STYLE)
# app.layout = dbc.Container(
#     [
#         html.H1("Cyber COVID"),
#         html.Hr()
#     ], fluid=True
# )
# if __name__ == "__main__":
#     app.run_server()
