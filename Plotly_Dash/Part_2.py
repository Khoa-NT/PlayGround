### ----------------------------------- Hello Dash ----------------------------------- ###

# # -*- coding: utf-8 -*-
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),
#
#     html.Div(children='''
#         Dash: A web application framework for Python.
#     '''),
#
#     dcc.Graph(
#         id='example-graph',
#         figure={
#             'data': [
#                 {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
#                 {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
#             ],
#             'layout': {
#                 'title': 'Dash Data Visualization'
#             }
#         }
#     )
# ])
# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', debug=True, port=8051)


### ----------------------------------- Color Hello Dash ----------------------------------- ###


# # -*- coding: utf-8 -*-
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# colors = {
#     'background': '#111111',
#     'text': '#7FDBFF'
# }
#
# app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
#     html.H1(
#         children='Hello Dash',
#         style={
#             'textAlign': 'center',
#             'color': colors['text']
#         }
#     ),
#
#     html.Div(children='Dash: A web application framework for Python.', style={
#         'textAlign': 'center',
#         'color': colors['text']
#     }),
#
#     dcc.Graph(
#         id='example-graph-2',
#         figure={
#             'data': [
#                 {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
#                 {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
#             ],
#             'layout': {
#                 'plot_bgcolor': colors['background'],
#                 'paper_bgcolor': colors['background'],
#                 'font': {
#                     'color': colors['text']
#                 }
#             }
#         }
#     )
# ])
#
# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', debug=True, port=8051)



### ----------------------------------- Create Table from Pandas Dash ----------------------------------- ###

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import pandas as pd
#
# df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')
#
#
# def generate_table(dataframe, max_rows=10):
#     return html.Table([
#         html.Thead(
#             html.Tr([html.Th(col) for col in dataframe.columns])
#         ),
#         html.Tbody([
#             html.Tr([
#                 html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
#             ]) for i in range(min(len(dataframe), max_rows))
#         ])
#     ])
#
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# app.layout = html.Div(children=[
#     html.H4(children='US Agriculture Exports (2011)'),
#     generate_table(df)
# ])
# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', debug=True, port=8051)


### ----------------------------------- More about Visualization ----------------------------------- ###
# https://plotly.com/python/
# https://plotly.com/javascript/plotlyjs-function-reference/#plotlyextendtraces

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import pandas as pd
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')
#
#
# app.layout = html.Div([
#     dcc.Graph(
#         id='life-exp-vs-gdp',
#         figure={
#             'data': [
#                 dict(
#                     x=df[df['continent'] == i]['gdp per capita'],
#                     y=df[df['continent'] == i]['life expectancy'],
#                     text=df[df['continent'] == i]['country'],
#                     mode='markers',
#                     opacity=0.7,
#                     marker={
#                         'size': 15,
#                         'line': {'width': 0.5, 'color': 'white'}
#                     },
#                     name=i
#                 ) for i in df.continent.unique()
#             ],
#             'layout': dict(
#                 xaxis={'type': 'log', 'title': 'GDP Per Capita'},
#                 yaxis={'title': 'Life Expectancy'},
#                 margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#                 legend={'x': 0, 'y': 1},
#                 hovermode='closest'
#             )
#         }
#     )
# ])
#
# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', debug=True, port=8051)

### ----------------------------------- Markdown ----------------------------------- ###

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# markdown_text = '''
# ### Dash and Markdown
#
# Dash apps can be written in Markdown.
# Dash uses the [CommonMark](http://commonmark.org/)
# specification of Markdown.
# Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
# if this is your first introduction to Markdown!
# '''
#
# app.layout = html.Div([
#     dcc.Markdown(children=markdown_text)
# ])
#
#
# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', debug=True, port=8051)

### ----------------------------------- Core Components ----------------------------------- ###

import colorsys
from random import shuffle


def color_generator(N, shuffle_color_list=True):
    HSV_tuples = [(x * 1.0 / N, 0.5, (x % 8) * 0.07 + 0.5) for x in range(N)]

    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    COLOR_LIST = [(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)) for rgb in RGB_tuples]
    if shuffle_color_list:
        shuffle(COLOR_LIST)
    COLOR_LIST[0] = (0, 0, 0)

    def index2rgb(index):
        return COLOR_LIST[index]

    def lbl2rgb(lbl):
        original_shape = np.squeeze(lbl).shape
        lbl = lbl.flatten().tolist()
        lbl = list(map(index2rgb, lbl))
        lbl = np.array(lbl).reshape(original_shape + (3,))
        return lbl

    return lbl2rgb

def checkInt2Tuple(input_num):
    if isinstance(input_num, int) or isinstance(input_num, float):
        output_tuple = (input_num, input_num)
    elif hasattr(input_num, '__len__') and (len(input_num) == 2):
        output_tuple = input_num
    else:
        raise ValueError(f"Input {input_num} doesn't have correct shape (Hy,Wx)")
    return output_tuple

def Pos2Coor(maxIndex, grid_size, grid_unit, k_size):
    list_coor = []
    for i in range(maxIndex):
        list_coor.append(Index2Coor(i, grid_size, grid_unit, k_size))
    return list_coor

def Index2Coor(act, grid_shape, grid_unit, k_size):
    grid_shape = checkInt2Tuple(grid_shape)
    grid_unit = checkInt2Tuple(grid_unit)
    k_size = checkInt2Tuple(k_size)
    # Row
    Hy = int((act // grid_shape[0]) * grid_unit[0] + k_size[0] / 2)
    # Col
    Wx = int((act % grid_shape[1]) * grid_unit[1] + k_size[1] / 2)
    return Hy, Wx

# https://dash.plotly.com/dash-core-components

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import dash_table

import math
import json
from datetime import datetime as dt
import pandas as pd
import io
import base64
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app = dash.Dash(__name__)

# https://community.plotly.com/t/dcc-tabs-filling-tabs-with-dynamic-content-how-to-organize-the-callbacks/6377/2
# app.config['suppress_callback_exceptions']=True
app.config.suppress_callback_exceptions = True


# Data Loading
@app.server.before_first_request
def load_all_footage():
    print("load_all_footage")

list_of_images = [
    'assets/horse.jpg',
    'assets/lbl.tif',
]
lbl2rgb_f = color_generator(600, True)


app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Tab 1', value='tab-1'),
        dcc.Tab(label='Tab 2', value='tab-2'),
        dcc.Tab(label='Tab 3', value='tab-3'),
        dcc.Tab(label='Tab 4', value='tab-4'),
        dcc.Tab(label='Tab 5', value='tab-5'),
        dcc.Tab(label='Tab 6', value='tab-6'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return tab1
    elif tab == 'tab-2':
        return tab2
    elif tab == 'tab-3':
        return tab3
    elif tab == 'tab-4':
        return tab4
    elif tab == 'tab-5':
        return tab5
    elif tab == 'tab-6':
        return tab6


# --------------- tab6 --------------- #
tab6_styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

tab6 = html.Div([
    dcc.Graph(
        id='basic-interactions',
        figure={
            'data': [
                {
                    'x': [1, 2, 3, 4],
                    'y': [4, 1, 3, 5],
                    'text': ['a', 'b', 'c', 'd'],
                    'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
                    'name': 'Trace 1',
                    'mode': 'markers',
                    'marker': {'size': 12}
                },
                {
                    'x': [1, 2, 3, 4],
                    'y': [9, 4, 1, 4],
                    'text': ['w', 'x', 'y', 'z'],
                    'customdata': ['c.w', 'c.x', 'c.y', 'c.z'],
                    'name': 'Trace 2',
                    'mode': 'markers',
                    'marker': {'size': 12}
                }
            ],
            'layout': {
                'clickmode': 'event+select'
            }
        }
    ),

    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown("""
                **Hover Data**

                Mouse over values in the graph.
            """),
            html.Pre(id='hover-data', style=tab6_styles['pre'])
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
            """),
            html.Pre(id='click-data', style=tab6_styles['pre']),
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Selection Data**

                Choose the lasso or rectangle tool in the graph's menu
                bar and then select points in the graph.

                Note that if `layout.clickmode = 'event+select'`, selection data also 
                accumulates (or un-accumulates) selected data if you hold down the shift
                button while clicking.
            """),
            html.Pre(id='selected-data', style=tab6_styles['pre']),
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Zoom and Relayout Data**

                Click and drag on the graph to zoom or click on the zoom
                buttons in the graph's menu bar.
                Clicking on legend items will also fire
                this event.
            """),
            html.Pre(id='relayout-data', style=tab6_styles['pre']),
        ], className='three columns')
    ])
])


@app.callback(
    Output('hover-data', 'children'),
    [Input('basic-interactions', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    Output('click-data', 'children'),
    [Input('basic-interactions', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)


@app.callback(
    Output('selected-data', 'children'),
    [Input('basic-interactions', 'selectedData')])
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)


@app.callback(
    Output('relayout-data', 'children'),
    [Input('basic-interactions', 'relayoutData')])
def display_relayout_data(relayoutData):
    return json.dumps(relayoutData, indent=2)
# --------------- tab5 --------------- #
tab5 = html.Div([
    html.Div(className='row', children=[html.H1("Button")], style = {
                     'width': '100%',
                     'display': 'flex',
                     'align-items': 'center',
                     'justify-content': 'center',
                 }),
    html.Div(dcc.Input(id='input-box', type='text')),
    html.Button('Submit', id='button'),
    html.Div(id='output-container-button',
             children='Enter a value and press submit'),
    html.Button('Button 1', id='btn-nclicks-1', n_clicks=0),
    html.Button('Button 2', id='btn-nclicks-2', n_clicks=0),
    html.Button('Button 3', id='btn-nclicks-3', n_clicks=0),
    html.Div(id='container-button-timestamp'),
    html.Button('Changing Name Button', id='btn-nclicks-4', n_clicks=0),
])

@app.callback(Output('btn-nclicks-4', 'children'),
              [Input('btn-nclicks-4', 'n_clicks'),])
def change_button_name(n_clicks):
    return f'clicked {n_clicks} time'


@app.callback(Output('container-button-timestamp', 'children'),
              [Input('btn-nclicks-1', 'n_clicks'),
               Input('btn-nclicks-2', 'n_clicks'),
               Input('btn-nclicks-3', 'n_clicks')])
def displayClick(btn1, btn2, btn3):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        msg = 'Button 1 was most recently clicked'
    elif 'btn-nclicks-2' in changed_id:
        msg = 'Button 2 was most recently clicked'
    elif 'btn-nclicks-3' in changed_id:
        msg = 'Button 3 was most recently clicked'
    else:
        msg = 'None of the buttons have been clicked yet'
    return html.Div(msg)

@app.callback(
    Output('output-container-button', 'children'),
    [Input('button', 'n_clicks')],
    [State('input-box', 'value')])
def update_output(n_clicks, value):
    return 'The input value was "{}" and the button has been clicked {} times'.format(
        value,
        n_clicks
    )

# --------------- tab4 --------------- #
tab4 = html.Div([
    html.H1("Upload YAML"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Label('YAML config:'),
    html.Pre(id='output-data-upload', style=tab6_styles['pre']),
    # html.Div(id='output-data-upload'),
])


# def parse_files_contents(contents, filename, date):
#     content_type, content_string = contents.split(',')
#
#     decoded = base64.b64decode(content_string)
#     try:
#         if 'csv' in filename:
#             # Assume that the user uploaded a CSV file
#             df = pd.read_csv(
#                 io.StringIO(decoded.decode('utf-8')))
#         elif 'xls' in filename:
#             # Assume that the user uploaded an excel file
#             df = pd.read_excel(io.BytesIO(decoded))
#     except Exception as e:
#         print(e)
#         return html.Div([
#             'There was an error processing this file.'
#         ])
#
#     return html.Div([
#         html.H5(filename),
#         html.H6(dt.fromtimestamp(date)),
#
#         dash_table.DataTable(
#             data=df.to_dict('records'),
#             columns=[{'name': i, 'id': i} for i in df.columns]
#         ),
#
#         html.Hr(),  # horizontal line
#
#         # For debugging, display the raw contents provided by the web browser
#         html.Div('Raw Content'),
#         html.Pre(contents[0:200] + '...', style={
#             'whiteSpace': 'pre-wrap',
#             'wordBreak': 'break-all'
#         })
#     ])

def parse_files_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    print(contents)
    print(content_type, content_string)
    print(filename)
    print(date)

    decoded = base64.b64decode(content_string)


    try:
        yaml = io.StringIO(decoded.decode('utf-8'))
        # yaml = io.BytesIO(decoded)
        conf = OmegaConf.load(yaml)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return conf.pretty()
    # return html.Textarea(conf.pretty(),
    #                      style={
    #                          'width': '50%',
    #                          'height': '500px',
    #                          'lineHeight': '30px',
    #                          'textAlign': 'left',
    #                          'margin': '1px'
    #                      })


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_files_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

# --------------- tab3 --------------- #
# https://dash.plotly.com/dash-core-components/upload

tab3 = html.Div([
    html.H1("Upload Image"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
    html.Hr(),
    html.Div([
        dcc.Dropdown(
            id='image-dropdown',
            options=[{'label': i, 'value': i} for i in list_of_images],
            # initially display the first entry in the list
            value=list_of_images[0]
        ),
        html.Img(id='image'),
    ]),
    html.Div([dcc.Graph(id="fig_img")]),
    html.Pre(id='coor_text', style=tab6_styles['pre'])

])

@app.callback(
    Output('coor_text', 'children'),
    [Input('fig_img', 'clickData')])
def display_click_data(clickData):
    ctx = dash.callback_context
    # print(clickData)
    # return json.dumps(clickData, indent=2)
    return json.dumps(
        {
            'states': ctx.states,
            'triggered': ctx.triggered,
            'inputs': ctx.inputs
        }, indent=2)


def parse_imgs_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(dt.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])
@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_imgs_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback([Output('image', 'src'),
               Output('fig_img', 'figure'),],
              [Input('image-dropdown', 'value')])
def update_image_src(image_path):
    img = np.array(Image.open(image_path), dtype="int32")

    # Create figure
    img_height, img_width = img.shape[0], img.shape[1]
    scale_factor = 1.0

    if '.tif' in image_path:
        img = lbl2rgb_f(img)

    img = Image.fromarray(img.astype("uint8"))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y = list_PosCoor[0],
            x = list_PosCoor[1],
            # x=[0, img_width * scale_factor],
            # y=[0, img_height * scale_factor],
            mode="markers",
            # marker_opacity=0
        )
    )
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )
    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )
    # # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img)
    )
    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )


    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()

    # print the image_path to confirm the selection is as expected
    print('current image_path = {}'.format(image_path))
    # encoded_image = base64.b64encode(open(image_path, 'rb').read())
    encoded_image = base64.b64encode(imgByteArr)
    return 'data:image/png;base64,{}'.format(encoded_image.decode()), fig

# --------------- tab1 --------------- #
tab1 = html.Div([

    dcc.Markdown('''
    #### Dash and Markdown

    Dash supports [Markdown](http://commonmark.org/help).

    Markdown is a simple way to write and format text.
    It includes a syntax for things like **bold text** and *italics*,
    [links](http://commonmark.org/help), inline `code` snippets, lists,
    quotes, and more.
    '''),

    dcc.RangeSlider(
        count=1,
        min=-5,
        max=10,
        step=0.5,
        value=[-3, 7]
    ),

    dcc.RangeSlider(
        marks={i: 'Label {}'.format(i) for i in range(-5, 7)},
        min=-5,
        max=6,
        value=[-3, 4]
    ),

    dcc.Textarea(
        placeholder='Enter a value...',
        value='This is a TextArea component',
        style={'width': '100%'}
    ),
    html.Button('Fig', id='btn-nclicks-5', n_clicks=0),
    html.Div([
        dcc.Graph(id="call_fig")
    ])

])
@app.callback(Output('call_fig', 'figure'),
              [Input('btn-nclicks-5', 'n_clicks'),])
def change_button_name(n_clicks):
    y = list_PosCoor[0]
    x = list_PosCoor[1]
    return px.scatter(x=x, y=y)

# --------------- tab2 --------------- #
# app.layout = html.Div([
tab2 = html.Div([
    ## test
    html.Label([
        'test_dropdown',
        dcc.Dropdown(
            options=[
                {'label': 'Khoa', 'value': 'NYC'},
                {'label': 'Khoa2', 'value': 'MTL'},
                {'label': 'Khoa3', 'value': 'SF'}
            ],
            value='MTL'
        ),
    ]),


    ## Dropdown
    html.Label('Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='MTL'
    ),

    ## Multi-Select Dropdown
    html.Label('Multi-Select Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value=['MTL', 'SF'],
        multi=True
    ),

    ## Radio Items
    html.Label('Radio Items'),
    dcc.RadioItems(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='MTL'
    ),

    ## Checkboxes
    html.Label('Checkboxes'),
    dcc.Checklist(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value=['MTL', 'SF']
    ),

    ## Text Input
    html.Label('Text Input'),
    dcc.Input(value='MTL', type='text'),

    ## Slider
    html.Label('Slider'),
    dcc.Slider(
        min=0,
        max=9,
        marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
        value=5,
    ),


    ## DatePickerSingle
    html.Label('DatePickerSingle'),
    dcc.DatePickerSingle(
        id='date-picker-single',
        date=dt(1997, 5, 10)
    ),

], style={'columnCount': 1})



if __name__ == '__main__':
    size = [336, 336]
    ksize = [65, 65]
    num_actions = 26
    grid_shape = (int(math.sqrt(num_actions - 1)),
                  int(math.sqrt(num_actions - 1)))

    grid_size = (size[0] - ksize[0],
                 size[1] - ksize[1])
    grid_unit = (grid_size[0] / (grid_shape[0] - 1),
                 grid_size[1] / (grid_shape[1] - 1))

    list_coor = []
    list_PosCoor = Pos2Coor(num_actions - 1,grid_shape, grid_unit, ksize)
    for i in range(0, len(list_PosCoor), grid_shape[0]):
        list_coor = list_PosCoor[i:i + grid_shape[0]] + list_coor
    list_PosCoor = np.array(list_coor).T
    app.run_server(host='0.0.0.0', debug=True, port=8051)


### ----------------------------------- Basic Dash Callbacks: Dash App Layout ----------------------------------- ###

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# app.layout = html.Div([
#     dcc.Input(id='my-id', value='initial value', type='text'),
#     html.Div(id='my-div')
# ])
#
#
# @app.callback(
#     Output(component_id='my-div', component_property='children'),
#     [Input(component_id='my-id', component_property='value')]
# )
# def update_output_div(input_value):
#     return 'You\'ve entered "{}"'.format(input_value)
#
# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', debug=True, port=8051)

### ----------------------------------- Basic Dash Callbacks: Dash App Layout 2 ----------------------------------- ###
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
#
# import pandas as pd
#
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# app.layout = html.Div([
#     dcc.Graph(id='graph-with-slider'),
#     dcc.Slider(
#         id='year-slider',
#         min=df['year'].min(),
#         max=df['year'].max(),
#         value=df['year'].min(),
#         marks={str(year): str(year) for year in df['year'].unique()},
#         step=None
#     )
# ])
#
#
# @app.callback(
#     Output('graph-with-slider', 'figure'),
#     [Input('year-slider', 'value')])
# def update_figure(selected_year):
#     filtered_df = df[df.year == selected_year]
#     traces = []
#     for i in filtered_df.continent.unique():
#         df_by_continent = filtered_df[filtered_df['continent'] == i]
#         traces.append(dict(
#             x=df_by_continent['gdpPercap'],
#             y=df_by_continent['lifeExp'],
#             text=df_by_continent['country'],
#             mode='markers',
#             opacity=0.7,
#             marker={
#                 'size': 15,
#                 'line': {'width': 0.5, 'color': 'white'}
#             },
#             name=i
#         ))
#
#     return {
#         'data': traces,
#         'layout': dict(
#             xaxis={'type': 'log', 'title': 'GDP Per Capita',
#                    'range':[2.3, 4.8]},
#             yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
#             margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#             legend={'x': 0, 'y': 1},
#             hovermode='closest',
#             transition = {'duration': 500},
#         )
#     }
#
# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', debug=True, port=8051)

### ----------------------------------- Basic Dash Callbacks: Dash App Layout 3 ----------------------------------- ###
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
#
# import pandas as pd
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')
#
# available_indicators = df['Indicator Name'].unique()
#
# app.layout = html.Div([
#     html.Div([
#
#         html.Div([
#             dcc.Dropdown(
#                 id='xaxis-column',
#                 options=[{'label': i, 'value': i} for i in available_indicators],
#                 value='Fertility rate, total (births per woman)'
#             ),
#             dcc.RadioItems(
#                 id='xaxis-type',
#                 options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
#                 value='Linear',
#                 labelStyle={'display': 'inline-block'}
#             )
#         ],
#         style={'width': '48%', 'display': 'inline-block'}),
#
#         html.Div([
#             dcc.Dropdown(
#                 id='yaxis-column',
#                 options=[{'label': i, 'value': i} for i in available_indicators],
#                 value='Life expectancy at birth, total (years)'
#             ),
#             dcc.RadioItems(
#                 id='yaxis-type',
#                 options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
#                 value='Linear',
#                 labelStyle={'display': 'inline-block'}
#             )
#         ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
#     ]),
#
#     dcc.Graph(id='indicator-graphic'),
#
#     dcc.Slider(
#         id='year--slider',
#         min=df['Year'].min(),
#         max=df['Year'].max(),
#         value=df['Year'].max(),
#         marks={str(year): str(year) for year in df['Year'].unique()},
#         step=None
#     )
# ])
#
# @app.callback(
#     Output('indicator-graphic', 'figure'),
#     [Input('xaxis-column', 'value'),
#      Input('yaxis-column', 'value'),
#      Input('xaxis-type', 'value'),
#      Input('yaxis-type', 'value'),
#      Input('year--slider', 'value')])
# def update_graph(xaxis_column_name, yaxis_column_name,
#                  xaxis_type, yaxis_type,
#                  year_value):
#     dff = df[df['Year'] == year_value]
#
#     return {
#         'data': [dict(
#             x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
#             y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
#             text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
#             mode='markers',
#             marker={
#                 'size': 15,
#                 'opacity': 0.5,
#                 'line': {'width': 0.5, 'color': 'white'}
#             }
#         )],
#         'layout': dict(
#             xaxis={
#                 'title': xaxis_column_name,
#                 'type': 'linear' if xaxis_type == 'Linear' else 'log'
#             },
#             yaxis={
#                 'title': yaxis_column_name,
#                 'type': 'linear' if yaxis_type == 'Linear' else 'log'
#             },
#             margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
#             hovermode='closest'
#         )
#     }



# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', debug=True, port=8051)

### ----------------------------------- Basic Dash Callbacks: Dash App Layout 4 ----------------------------------- ###
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# app.layout = html.Div([
#     dcc.Input(
#         id='num-multi',
#         type='number',
#         value=5
#     ),
#     html.Table([
#         html.Tr([html.Td(['x', html.Sup(2)]), html.Td(id='square')]),
#         html.Tr([html.Td(['x', html.Sup(3)]), html.Td(id='cube')]),
#         html.Tr([html.Td([2, html.Sup('x')]), html.Td(id='twos')]),
#         html.Tr([html.Td([3, html.Sup('x')]), html.Td(id='threes')]),
#         html.Tr([html.Td(['x', html.Sup('x')]), html.Td(id='x^x')]),
#     ]),
# ])
#
#
# @app.callback(
#     [Output('square', 'children'),
#      Output('cube', 'children'),
#      Output('twos', 'children'),
#      Output('threes', 'children'),
#      Output('x^x', 'children')],
#     [Input('num-multi', 'value')])
# def callback_a(x):
#     return x**2, x**3, 2**x, 3**x, x**x
#
# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', debug=True, port=8051)


### ----------------------------------- Basic Dash Callbacks: Dash App Layout 5 ----------------------------------- ###

# # -*- coding: utf-8 -*-
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# all_options = {
#     'America': ['New York City', 'San Francisco', 'Cincinnati'],
#     'Canada': [u'Montréal', 'Toronto', 'Ottawa']
# }
# app.layout = html.Div([
#     dcc.RadioItems(
#         id='countries-radio',
#         options=[{'label': k, 'value': k} for k in all_options.keys()],
#         value='America'
#     ),
#
#     html.Hr(),
#
#     dcc.RadioItems(id='cities-radio'),
#
#     html.Hr(),
#
#     html.Div(id='display-selected-values')
# ])
#
#
# @app.callback(
#     Output('cities-radio', 'options'),
#     [Input('countries-radio', 'value')])
# def set_cities_options(selected_country):
#     return [{'label': i, 'value': i} for i in all_options[selected_country]]
#
#
# @app.callback(
#     Output('cities-radio', 'value'),
#     [Input('cities-radio', 'options')])
# def set_cities_value(available_options):
#     return available_options[0]['value']
#
#
# @app.callback(
#     Output('display-selected-values', 'children'),
#     [Input('countries-radio', 'value'),
#      Input('cities-radio', 'value')])
# def set_display_children(selected_country, selected_city):
#     return u'{} is a city in {}'.format(
#         selected_city, selected_country,
#     )
#
# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', debug=True, port=8051)