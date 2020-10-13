import json
import time
import pickle
import random

from flask import (Flask, jsonify, render_template)
import requests
import gensim
from pymongo import MongoClient
from fake_useragent import UserAgent
import numpy as np
import dgl
import torch
import torch.nn.functional as F

from graph.models import SAGE

app = Flask(__name__)
ua = UserAgent()
with open('./data/semantic_scholar/ss_paper_ids_v4.pkl', 'rb') as f:
    ss_paper_ids = pickle.load(f)
ss_paper_ids = set(ss_paper_ids)

with open('./data/semantic_scholar/semantic_scholar_node_mapping_v4.pkl',
          'rb') as f:
    node_mapping = pickle.load(f)

with open(
        './data/semantic_scholar/semantic_scholar_node_mapping_reversed_v4.pkl',
        'rb') as f:
    reversed_node_mapping = pickle.load(f)

# connect to mongo
client = MongoClient('localhost', 27017)
db = client['paper']
col_v4 = db['semantic_scholar_v4']

# load graph
g, _ = dgl.load_graphs('./data/semantic_scholar/g_v4.bin')
g = g[0]

# load graphsage model
device = 'cpu'
in_feats = 300
batch_size = 128
model = SAGE(in_feats, 128, 32, 2, F.relu, 0.2)
model = model.to(device)
model.load_state_dict(torch.load('./models/semantic_scholar/ss_v4_model.pth'))
model.eval()

# load doc2vec model for node features
doc2vec = gensim.models.Doc2Vec.load('./models/doc2vec/doc2vec.bin')
start_alpha = 0.01
infer_epoch = 1000


@app.route('/')
def test():

    links = [{
        "source": "Microsoft",
        "target": "Amazon",
        "type": "licensing"
    }, {
        "source": "Microsoft",
        "target": "HTC",
        "type": "licensing"
    }, {
        "source": "Samsung",
        "target": "Apple",
        "type": "suit"
    }, {
        "source": "Motorola",
        "target": "Apple",
        "type": "suit"
    }, {
        "source": "Nokia",
        "target": "Apple",
        "type": "resolved"
    }, {
        "source": "HTC",
        "target": "Apple",
        "type": "suit"
    }, {
        "source": "Kodak",
        "target": "Apple",
        "type": "suit"
    }, {
        "source": "Microsoft",
        "target": "Barnes & Noble",
        "type": "suit"
    }, {
        "source": "Microsoft",
        "target": "Foxconn",
        "type": "suit"
    }, {
        "source": "Oracle",
        "target": "Google",
        "type": "suit"
    }, {
        "source": "Apple",
        "target": "HTC",
        "type": "suit"
    }, {
        "source": "Microsoft",
        "target": "Inventec",
        "type": "suit"
    }, {
        "source": "Samsung",
        "target": "Kodak",
        "type": "resolved"
    }, {
        "source": "LG",
        "target": "Kodak",
        "type": "resolved"
    }, {
        "source": "RIM",
        "target": "Kodak",
        "type": "suit"
    }, {
        "source": "Sony",
        "target": "LG",
        "type": "suit"
    }, {
        "source": "Kodak",
        "target": "LG",
        "type": "resolved"
    }, {
        "source": "Apple",
        "target": "Nokia",
        "type": "resolved"
    }, {
        "source": "Qualcomm",
        "target": "Nokia",
        "type": "resolved"
    }, {
        "source": "Apple",
        "target": "Motorola",
        "type": "suit"
    }, {
        "source": "Microsoft",
        "target": "Motorola",
        "type": "suit"
    }, {
        "source": "Motorola",
        "target": "Microsoft",
        "type": "suit"
    }, {
        "source": "Huawei",
        "target": "ZTE",
        "type": "suit"
    }, {
        "source": "Ericsson",
        "target": "ZTE",
        "type": "suit"
    }, {
        "source": "Kodak",
        "target": "Samsung",
        "type": "resolved"
    }, {
        "source": "Apple",
        "target": "Samsung",
        "type": "suit"
    }, {
        "source": "Kodak",
        "target": "RIM",
        "type": "suit"
    }, {
        "source": "Nokia",
        "target": "Qualcomm",
        "type": "suit"
    }]

    return render_template('index.html', links=links)


def infer_unseen_node(ss_paper_id, topk):

    start = time.time()

    test_url = 'https://api.semanticscholar.org/v1/paper/{}'.format(
        ss_paper_id)
    header = {'User-Agent': str(ua.random)}
    req = requests.get(test_url, headers=header)
    json_data = req.json()

    in_papers = json_data['citations']
    out_papers = json_data['references']
    abstract = json_data['abstract']

    # get paper's id
    in_paper_ids = [paper['paperId'] for paper in in_papers]
    out_paper_ids = [paper['paperId'] for paper in out_papers]

    # check exist in database
    in_paper_ids = [id_ for id_ in in_paper_ids if id_ in ss_paper_ids]
    out_paper_ids = [id_ for id_ in out_paper_ids if id_ in ss_paper_ids]

    # mapping to origin ss paper's id
    ss_in_ids = [node_mapping[i] for i in in_paper_ids]
    ss_out_ids = [node_mapping[i] for i in out_paper_ids]
    ss_dgl_ids = ss_in_ids + ss_out_ids

    first_subgraph = g.subgraph(torch.tensor(ss_dgl_ids))
    print(">>> 1st graph before adding new edges - {}s".format(time.time() -
                                                               start))
    print(first_subgraph)

    id_subgraph2graph = {
        int(old_label): int(new_label)
        for new_label, old_label in zip(ss_dgl_ids, first_subgraph.nodes())
    }
    id_graph2subgraph = dict(map(reversed, id_subgraph2graph.items()))

    no_nodes = len(first_subgraph.nodes())
    # add new edges for unseen node
    _src_subgraph_node_ids1 = [id_graph2subgraph[i] for i in ss_in_ids]
    _dst_subgraph_node_ids1 = [no_nodes] * len(_src_subgraph_node_ids1)

    _dst_subgraph_node_ids2 = [id_graph2subgraph[i] for i in ss_out_ids]
    _src_subgraph_node_ids2 = [no_nodes] * len(_dst_subgraph_node_ids2)

    _src_nodes = _src_subgraph_node_ids1 + _src_subgraph_node_ids2
    _dst_nodes = _dst_subgraph_node_ids1 + _dst_subgraph_node_ids2

    first_subgraph.add_edges(_src_nodes, _dst_nodes)
    first_subgraph.add_edges(_dst_nodes, _src_nodes)
    print(">>> 1st graph after adding new node - {}s".format(time.time() -
                                                             start))
    print(first_subgraph)

    # add node feature for new unseen node
    infer_vec = doc2vec.infer_vector(abstract.lower().strip().split(),
                                     alpha=start_alpha,
                                     steps=infer_epoch).astype(np.float32)
    first_subgraph.ndata['features'][-1] = torch.tensor(infer_vec)

    # make inference on unseen node
    # WARNING: set model to eval mode
    y_pred = model.inference(first_subgraph, first_subgraph.ndata['features'],
                             batch_size, device)
    y_pred = y_pred.detach().numpy()

    # get top-k most similarity
    dist_ids = np.linalg.norm(y_pred - y_pred[-1], axis=1)
    y_pred_ids = np.argsort(dist_ids)[1:topk + 1]
    dist_ids = dist_ids[y_pred_ids]

    # make 2nd subgraph from top-k
    second_subgraph = first_subgraph.subgraph(
        torch.tensor(y_pred_ids.tolist() +
                     [first_subgraph.number_of_nodes() - 1]))
    print('>>> 2nd graph - topk similarity - {}s'.format(time.time() - start))
    print(second_subgraph)

    # visualize
    # id_subgraph2graph = {
    #     int(old_label): int(new_label) for new_label, old_label in zip(ss_dgl_ids, second_subgraph.nodes())
    # }
    # vis_graph(second_subgraph, name=mapping)
    # TODO: add distance measure the similarity of each pair node
    # vis_graph(second_subgraph)
    # plt.savefig('./assets/foo.png')

    # result_ss_ids = [reversed_node_mapping[j] for j in [id_subgraph2graph[i] for i in y_pred_ids]]
    # for paper in col_v4.find({"id": {"$in": result_ss_ids}}):
    #     print(paper['title'])
    #     print('>' * 50)
 
    # nodes = []
    # for _node in second_subgraph.nodes().cpu().numpy():
    #     nodes.append({'id': int(_node), 'group': random.choice(range(3))})

    links = []
    _src_ids = second_subgraph.edges()[0].cpu().numpy()
    _dst_ids = second_subgraph.edges()[1].cpu().numpy()
    for _src, _dst in zip(_src_ids, _dst_ids):
        links.append({
            'source': int(_src),
            'target': int(_dst),
            'value': random.choice(range(1, 4))
        })

    d3_json = {
        'time': time.time() - start,
        # 'nodes': nodes,
        'links': links,
        'is_existed': False
    }
    print(">>> Pipeline: {}s".format(time.time() - start))

    return d3_json


def infer_existed_node(paper, topk):

    start = time.time()

    paper_id = paper['id']
    in_paper_ids = paper['inCitations']
    out_paper_ids = paper['outCitations']
    abstract = paper['paperAbstract']

    # mapping to origin ss paper's id
    ss_origin_id = node_mapping[paper_id]
    ss_in_ids = [node_mapping[i] for i in in_paper_ids]
    ss_out_ids = [node_mapping[i] for i in out_paper_ids]
    ss_dgl_ids = ss_in_ids + ss_out_ids + [ss_origin_id]

    # ?? make k-hop subgraph, default 1-hop ??

    # make 1st subgraph
    first_subgraph = g.subgraph(torch.tensor(ss_dgl_ids))
    print(">>> 1st graph before adding new edges - {}s".format(time.time() -
                                                               start))
    print(first_subgraph)

    id_subgraph2graph = {
        int(old_label): int(new_label)
        for new_label, old_label in zip(ss_dgl_ids, first_subgraph.nodes())
    }
    id_graph2subgraph = dict(map(reversed, id_subgraph2graph.items()))

    no_nodes = len(first_subgraph.nodes())
    # add new edges for unseen node
    _src_subgraph_node_ids1 = [id_graph2subgraph[i] for i in ss_in_ids]
    _dst_subgraph_node_ids1 = [no_nodes] * len(_src_subgraph_node_ids1)

    _dst_subgraph_node_ids2 = [id_graph2subgraph[i] for i in ss_out_ids]
    _src_subgraph_node_ids2 = [no_nodes] * len(_dst_subgraph_node_ids2)

    _src_nodes = _src_subgraph_node_ids1 + _src_subgraph_node_ids2
    _dst_nodes = _dst_subgraph_node_ids1 + _dst_subgraph_node_ids2

    first_subgraph.add_edges(_src_nodes, _dst_nodes)
    first_subgraph.add_edges(_dst_nodes, _src_nodes)
    print(">>> 1st graph after adding new node - {}s".format(time.time() -
                                                             start))
    print(first_subgraph)

    # ??
    origin_node_feature = g.ndata['features'][node_mapping[paper_id]]
    first_subgraph.ndata['features'][-1] = origin_node_feature

    # make inference on unseen node
    # WARNING: set model to eval mode
    y_pred = model.inference(first_subgraph, first_subgraph.ndata['features'],
                             batch_size, device)
    y_pred = y_pred.detach().numpy()

    # get top-k most similarity
    dist_ids = np.linalg.norm(y_pred - y_pred[-1], axis=1)
    y_pred_ids = np.argsort(dist_ids)[1:topk + 1]
    dist_ids = dist_ids[y_pred_ids]

    # make 2nd subgraph from top-k
    second_subgraph = first_subgraph.subgraph(
        torch.tensor(y_pred_ids.tolist() +
                     [first_subgraph.number_of_nodes() - 1]))
    print('>>> 2nd graph - topk similarity - {}s'.format(time.time() - start))
    print(second_subgraph)

    # TODO: add distance measure the similarity of each pair node

    # nodes = []
    # for _node in second_subgraph.nodes().cpu().numpy():
    #     nodes.append({'id': int(_node), 'group': random.choice(range(3))})

    links = []
    _src_ids = second_subgraph.edges()[0].cpu().numpy()
    _dst_ids = second_subgraph.edges()[1].cpu().numpy()
    for _src, _dst in zip(_src_ids, _dst_ids):
        links.append({
            'source': int(_src),
            'target': int(_dst),
            'value': random.choice(range(1, 4))
        })

    d3_json = {
        'time': time.time() - start,
        # 'nodes': nodes,
        'links': links,
        'is_existed': True
    }
    # with open('./data/ss_json/{}.json'.format(paper_id), 'w') as outfile:
    #     json.dump(d3_json, outfile)

    print(">>> Pipeline: {}s".format(time.time() - start))

    return d3_json


def convert_format(json_result):
    result = []

    links = json_result['links']
    for link in links:
        result.append({"source": link['source'], "target": link['target']})

    return result


@app.route('/<ss_paper_id>/')
def serve(ss_paper_id):

    print('>' * 50)
    topk = 20
    # check paper exists
    paper = list(col_v4.find({'id': ss_paper_id}))

    if len(paper) == 0:
        # unseen node
        # sample: 962dc29fdc3fbdc5930a10aba114050b82fe5a3e - detr
        json_result = infer_unseen_node(ss_paper_id, topk)
    else:
        # existed node
        # sample: 6b7d6e6416343b2a122f8416e69059ce919026ef - graphsage
        json_result = infer_existed_node(paper[0], topk)

    # return jsonify(json_result)
    links = convert_format(json_result)
    return render_template('index.html', links=links)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
