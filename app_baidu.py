import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import time
import torch
import torch.nn as nn
from GPT2 import GPT2Model, GPT2Tokenizer
import os
import numpy as np
from tqdm import tqdm


# os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = 'cuda' #'cuda'


def similarity_vector_matrix(arr, brr):
    return arr.dot(brr.T) / (np.sqrt(np.sum(arr*arr)) * np.sqrt(np.sum(brr*brr, axis=1)))


def tokenize_input(inputStr, tokenizer, seq_length=1024):
    pad_id = tokenizer.encoder['<pad>']
    tokenized_sentence = tokenizer.encode(inputStr)[:seq_length-20]
    tokens = tokenized_sentence
    token_length = len(tokens)
    tokens.extend([pad_id] * (seq_length - token_length))
    tokens = torch.tensor(tokens, dtype=torch.long)
    return tokens.reshape(1,1024), [token_length]


def stock_search(query, topk=10):

    tokens, token_length = tokenize_input(query, tokenizer, seq_length=1024)
    output = model(tokens.to(device))
    vector = output[0,token_length[0]].detach().cpu().numpy()

    res = similarity_vector_matrix(vector, vectors)
    idxs = np.argsort(res)[::-1]

    topk_idxs = idxs[:topk]
    names = [ticker_names[idx] for idx in topk_idxs]
    infos = [ticker_infos[idx] for idx in topk_idxs]
    ids = [ticker_ids[idx] for idx in topk_idxs]
    # print(names)
    return names, ids, infos


tokenizer = GPT2Tokenizer(
    'GPT2/bpe/vocab.json',
    'GPT2/bpe/chinese_vocab.model',
    max_len=512)
    
model = GPT2Model(
    vocab_size=30000,
    layer_size=12,
    block_size=1024,
    embedding_dropout=0.0,
    embedding_size=768,
    num_attention_heads=12,
    attention_dropout=0.0,
    residual_dropout=0.0
)

state_dict = torch.load('../models/model_pretrain_distill.pth', map_location='cpu')
model.load_state_dict(state_dict)

model.eval()

model.to(device)

print('loaded success')


data = np.load('./data_ignore/vectors_gpt2.pkl', allow_pickle=True)


ticker_names = []
ticker_ids = []
vectors = []
ticker_infos = []

for idx in tqdm(range(len(data))):
    if 'vector' in data[idx].keys():
        vectors.append(data[idx]['vector'])
        ticker_ids.append(data[idx]['ticker_id'])
        ticker_names.append(data[idx]['ticker_name'])
        ticker_infos.append(data[idx]['info'])

vectors = np.array(vectors)
#--------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
)

welcome = dbc.Container(
    html.H1('Welcome'),
    style={
        'margin-top': '120px',
        'max-width': '200px'
    }
)

app.layout = html.Div(
    [
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),

        dbc.Container(
            [
                html.Img(
                    src='assets/asklora_logo.svg',
                    style={
                        'width': '50%',
                        'height': '50%',
                        'margin-left': '170px',
                        }
                ),

                html.Hr(),  # 分割线

                # 输入控件
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label('QUERY :'),
                                dbc.Input(type="text", id='input-query'),
                            ]
                        )
                    ]
                ),

                html.Br(),  # 空格

                html.Hr(),

                # 提交查询按钮

                dbc.Button(
                    'Query',
                    id='query-button',
                    size='lg',
                    color="primary",
                    className="d-grid gap-4 col-4 ",
                    style={'margin-left': '250px',}
                ),

                html.Hr(),
                dbc.Spinner(html.Div(id="loading-query")),

                dbc.Tabs(
                    id='output-value',
                    style={'margin-top': '20px'},
                )

            ],
            style={
            'margin-top': '-20px',
            'max-width': '800px'
            }
        ),

    ]
)


@app.callback(
    Output("loading-query", "children"),
    Output('output-value', 'children'),
    Input('query-button', 'n_clicks'),
    State('input-query', 'value'),
    prevent_initial_call=True,
)
def render_content(n_clicks, query):
    '''
    根据用户控件输入结果，进行相应查询结果的渲染
    :param n_clicks: 查询按钮点击次数
    :param ticker_id: 已选择的车辆对应id
    :return:
    '''

    # 当按钮被新一轮点击后
    if n_clicks:

        time.sleep(0.5)

        if query:

            info_alert = 'query success'

            names, ids, infos = stock_search(query, topk=15)

            # 初始化Tabs返回结果
            tabs = []
            for name, id, info in zip(names, ids, infos):

                tabs.append(
                    dbc.Tab(
                        html.Blockquote(
                            [
                                html.Br(),
                                html.H4(f"{name} {id}"),
                                html.P(info),
                                html.Br(),
                            ],
                            style={
                                'background-color': 'rgba(211, 211, 211, 0.25)',
                                'text-indent': '1rem'
                            }
                        ),
                        label=name,
                    ),
                ),

            # 返回渲染结果
            return dbc.Alert(info_alert, color="success" ,style={'margin-top': '0px',}), tabs

    return dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)
