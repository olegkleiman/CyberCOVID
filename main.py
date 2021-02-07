import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import plotly.graph_objs as go

import requests
import io
import pandas as pd
import numpy as np

from enumerated_dates import EnumeratedDates
from city_model import CityModel

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

url = "https://raw.githubusercontent.com/idandrd/israel-covid19-data/master/CityData.csv"
s = requests.get(url).content

df = pd.read_csv(io.StringIO(s.decode('utf-8')))

# 2a. Throw out nan values and other mess from the downloaded DataFrame.
#   Thanks to God, Pandas has built-in functions for such a purpose
# Output of Step 2a: df
df.fillna(0, inplace=True)  # prefer mutable versions because it modifies other views
df.replace('-', 0., inplace=True)

#
# 2b. Prepare the dates obtained from the DataFrame to participate in regression:
#   just enumerate all the dates, i.e. convert them to the running number
# Output of Step 2b: enumeratedDates array

keys = df.keys()
dates = EnumeratedDates(keys[2:])

city_models = []
for i in range(df['City'].count()):
    city_models.append({
        "name": df['City'][i],
        "rowId": i
    })

#     [{
#     "name" : "Jerusalem",
#     "rowId": 0
# },
# {
#     "name": "Tel-Aviv",
#     "rowId": 1
# }]

# city_names = df.loc[:, "City"]

app.layout = html.Div(children=[
    html.H1(children='CyberCOVID'),
    html.Div([
        dcc.Dropdown(
            id='city-name-dropdown',
            options=[
                {"label": x['name'], "value": x['rowId']} for x in city_models
            ]
        ),
        html.Div(id='dd-output-container'),
        dcc.Graph(id='city-graph')
    ])
])


@app.callback(
    dash.dependencies.Output(component_id='city-graph', component_property='figure'),
    [dash.dependencies.Input('city-name-dropdown', 'value')]
)
def city_changed(row_id):
    # list comprehension
    # city_model = next((x for x in city_models if x['name'] == city_name), None)

    fig = go.Figure(layout=go.Layout(height=400, width=1200))

    try:
        row = df.iloc[row_id].to_numpy()
    except TypeError:
        return fig
    else:
        model = CityModel(city_name=row[0], x=dates.indices, y=row[2:])

        fig.layout.title = row[0]
        model.display(figure=fig, regressors=dates)
        return fig

    # return 'You have selected "{}"'.format(city_model['name'] if city_model is not None else 'None')


if __name__ == '__main__':
    app.run_server(debug=True)
