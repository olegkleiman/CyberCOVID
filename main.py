# CyberCOVID project.

# Project aims: TODO

import io
import numpy as np
import pandas as pd
import requests

import platform
print('Python platform: {}'.format(platform.architecture()[0]))

# Flask stuff
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# import dash_bootstrap_components as dbc

from city_model import CityModel
from enumerated_dates import EnumeratedDates
from regressions import calculate_regression_params, calc, SimpleLinearRegression
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
df.fillna(0, inplace=True) # prefer mutable versions because it modifies other views
df.replace('-', 0., inplace=True)
# print(df)

#
# 2b. Prepare the dates obtained from the DataFrame to participate in regression:
#   just enumerate all the dates, i.e. convert them to the running number
# Output of Step 2b: enumeratedDates array

keys = df.keys()
dates = EnumeratedDates(keys[2:])


#
# 3. Calculate the regression model parameters.

# The Linear Regression models implemented here (SkLearn and TensorFlow) accept numpy arrays.
# Hence, we convert the prepared DataFrame into numpy array (and intentionally losing the captions raw)
ndata = df.to_numpy()
city_models = np.array([])

# The function 'calculate_regression_params' below will be called for each city (row) in the prepared dataset

for row in ndata:
    # First two columns are defined in the original DataFrame as 'City' and 'Population'. We skip them for the data

    # SkLearn
    model = calculate_regression_params(x=dates.indices, y=row[2:], name=row[0])

    # Uncomment for TenforFlow based model
    # linear_model = SimpleLinearRegression('zeros')
    # linear_model.train(dates.indices, row[2:], learning_rate=0.1, epochs=50)

    city_models = np.append(city_models, model)


# _models = np.apply_along_axis(calc, 1, ndata)

# Just show some regressions for largest cities
# if the model's score is acceptable according to score
THRESHOLD = 0.9
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
        if city_models[i].score > THRESHOLD:
            # %%
            city_models[i].display(dates.labels)
            # %%

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
