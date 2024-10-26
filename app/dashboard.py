import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from flask import Flask, render_template

# Initialize Flask app
server = Flask(__name__)

# Load cleaned data
df = pd.read_csv('data/cleaned_data.csv')

# Initialize Dash app
app = dash.Dash(__name__, server=server, routes_pathname_prefix='/dashboard/')


# Layout of the dashboard
app.layout = html.Div(children=[
    html.H1(children='Sentiment Analysis Dashboard'),
    dcc.Graph(
        id='sentiment-graph',
        figure = px.histogram(df, x='Sentiment', title='Sentiment Distribution')
    ),
    html.Div(children=[
        dcc.Input(id='input-text', type='text', placeholder='Enter text for prediction'),
        html.Button('Submit', id='submit-button'),
        html.Div(id='output-div')
    ])
])

@app.callback(
    dash.dependencies.Output('output-div', 'children'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks is None or value is None:
        return ''
    
    # Make API call for prediction
    import requests
    response = requests.post('http://127.0.0.1:5000/api/predict', json={'text': value})
    if response.status_code == 200:
        predicted_class = response.json().get('predicted_class')
        return f'Predicted Sentiment Class: {predicted_class}'
    return 'Error in prediction'

if __name__ == '__main__':
    app.run(debug=True)
