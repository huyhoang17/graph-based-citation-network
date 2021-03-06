import gc
import time
import random
import pickle
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for
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
with open("./data/semantic_scholar/ss_paper_ids_v4.pkl", "rb") as f:
    ss_paper_ids = pickle.load(f)
    ss_paper_ids = set(ss_paper_ids)

with open("./data/semantic_scholar/semantic_scholar_node_mapping_v4.pkl", "rb") as f:
    node_mapping = pickle.load(f)
    reversed_node_mapping = dict(map(reversed, node_mapping.items()))

with open("./data/semantic_scholar/top200.pkl", "rb") as f:
    top200_ids = pickle.load(f)

# connect to mongo
client = MongoClient("localhost", 27017)
db = client["paper"]
col_v4 = db["semantic_scholar_v4"]

# load graph
topk = 20
device = "cpu"
g, _ = dgl.load_graphs("./data/semantic_scholar/g_v4.bin")
g = g[0]
g.to(device)

# load graphsage model
in_feats = 300
batch_size = 128
model = SAGE(in_feats, 128, 32, 2, F.relu, 0.2)
model = model.to(device)
model.load_state_dict(
    torch.load("./models/semantic_scholar/ss_v4_model.pth", map_location=device)
)
model.eval()

# load doc2vec model for node features
doc2vec = gensim.models.Doc2Vec.load("./models/doc2vec/doc2vec.bin")
start_alpha = 0.01
infer_epoch = 1000


# TODO
# 1. define k-hop subgraph
# 2. redirect url


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(">>> Function {}: {}'s".format(func.__name__, end - start))
        return result

    return wrapper


def make_khop_ids(g, ss_dgl_ids):

    nids = []
    for ss_id in ss_dgl_ids:
        nid = g.in_edges(ss_id)[0].unique()
        nids.append(nid)

    nids = torch.cat(nids).unique()
    return nids


def make_sample_nodes(sample_size=10):
    papers = []

    random_ids = random.sample(top200_ids, sample_size)
    for paper in col_v4.find({"id": {"$in": random_ids}}):
        papers.append(
            {
                "id": paper["id"],
                "title": paper["title"],
                "abstract": paper["paperAbstract"],
                "ss_url": paper["s2Url"],
            }
        )

    return papers


def make_request_json(url):

    if "semanticscholar.org" in url:
        paper_id = url.strip("/").split("/")[-1]
        if len(paper_id) != 40:
            return None

        ss_api_url = "https://api.semanticscholar.org/v1/paper/{}".format(paper_id)
    elif "arxiv.org" in url:
        paper_id = url.strip("/").split("/")[-1]
        if paper_id.endswith(".pdf"):
            paper_id = paper_id[:-4]

        # paper's version
        # sample: xxxx.xxxxxv4
        v_index = paper_id.find("v")
        if v_index != -1:
            paper_id = paper_id[:v_index]

        ss_api_url = "https://api.semanticscholar.org/v1/paper/arXiv:{}".format(
            paper_id
        )
    elif "aclweb.org" in url:
        paper_id = url.strip("/").split("/")[-1]
        if paper_id.endswith(".pdf"):
            paper_id = paper_id[:-4]

        ss_api_url = "https://api.semanticscholar.org/v1/paper/ACL:{}".format(paper_id)
    else:
        ss_api_url = None

    if ss_api_url is not None:
        # make request
        try:
            header = {"User-Agent": str(ua.random)}
            req = requests.get(ss_api_url, headers=header)
            json_data = req.json()
            return json_data
        except Exception as e:
            print(e)
            return None
    else:
        return None


@timer
def infer_unseen_node(json_data, topk):

    start = time.time()

    in_papers = json_data["citations"]
    out_papers = json_data["references"]
    abstract = json_data["abstract"]
    title_paper = json_data["title"]
    paper_url = json_data["url"]

    # get paper's id
    in_paper_ids = [paper["paperId"] for paper in in_papers]
    out_paper_ids = [paper["paperId"] for paper in out_papers]

    # check exist in database
    in_paper_ids = [id_ for id_ in in_paper_ids if id_ in ss_paper_ids]
    out_paper_ids = [id_ for id_ in out_paper_ids if id_ in ss_paper_ids]

    # mapping to origin ss paper's id
    ss_in_ids = [node_mapping[i] for i in in_paper_ids]
    ss_out_ids = [node_mapping[i] for i in out_paper_ids]
    ss_dgl_ids = ss_in_ids + ss_out_ids
    ss_dgl_ids = list(set(ss_dgl_ids))

    # make k-hop graph??

    first_subgraph = g.subgraph(torch.tensor(ss_dgl_ids))
    print(">>> 1st graph before adding new edges - {}s".format(time.time() - start))
    print(first_subgraph)

    id_subgraph2graph = {
        int(old_label): int(new_label)
        for new_label, old_label in zip(ss_dgl_ids, first_subgraph.nodes())
    }
    id_graph2subgraph = dict(map(reversed, id_subgraph2graph.items()))

    no_nodes = first_subgraph.number_of_nodes()
    # add new edges for unseen node
    _src_subgraph_node_ids1 = [id_graph2subgraph[i] for i in ss_in_ids]
    _dst_subgraph_node_ids1 = [no_nodes] * len(_src_subgraph_node_ids1)

    _dst_subgraph_node_ids2 = [id_graph2subgraph[i] for i in ss_out_ids]
    _src_subgraph_node_ids2 = [no_nodes] * len(_dst_subgraph_node_ids2)

    _src_nodes = _src_subgraph_node_ids1 + _src_subgraph_node_ids2
    _dst_nodes = _dst_subgraph_node_ids1 + _dst_subgraph_node_ids2

    first_subgraph.add_edges(_src_nodes, _dst_nodes)
    first_subgraph.add_edges(_dst_nodes, _src_nodes)
    print(">>> 1st graph after adding new node - {}s".format(time.time() - start))
    print(first_subgraph)

    # add node feature for new unseen node
    infer_vec = doc2vec.infer_vector(
        abstract.lower().strip().split(), alpha=start_alpha, steps=infer_epoch
    ).astype(np.float32)
    first_subgraph.ndata["features"][-1] = torch.tensor(infer_vec)

    # make inference on unseen node
    # WARNING: set model to eval mode
    y_pred = model.inference(
        first_subgraph, first_subgraph.ndata["features"], batch_size, device
    )
    y_pred = y_pred.detach().numpy()

    no_edges = first_subgraph.number_of_edges()
    degree = no_edges / no_nodes

    if degree < 30:
        topk = int(topk * 1.5)

    # get top-k most similarity
    dist_ids = np.linalg.norm(y_pred - y_pred[-1], axis=1)
    y_pred_ids = np.argsort(dist_ids)[1 : topk + 1]
    dist_ids = dist_ids[y_pred_ids]

    # make 2nd subgraph from top-k
    second_subgraph = first_subgraph.subgraph(
        torch.tensor(y_pred_ids.tolist() + [first_subgraph.number_of_nodes() - 1])
    )
    print(">>> 2nd graph - topk similarity - {}s".format(time.time() - start))
    print(second_subgraph)

    paper_ids = [
        reversed_node_mapping[k]
        for k in [id_subgraph2graph[j] for j in [i for i in y_pred_ids]]
    ]
    # https://stackoverflow.com/a/40048951
    result_papers = list(col_v4.find({"id": {"$in": paper_ids}}))
    result_papers.sort(key=lambda thing: paper_ids.index(thing["id"]))
    for index, paper in enumerate(result_papers):
        paper["dist"] = round(dist_ids[index], 4)
    result_papers.append(
        {"title": title_paper, "s2Url": paper_url, "paperAbstract": abstract}
    )
    print(">>> Mongo query - {}s".format(time.time() - start))

    # retrieve node for d3 visualize
    links = []
    _src_titles = [
        result_papers[i]["title"] for i in second_subgraph.edges()[0].cpu().numpy()
    ]
    _dst_titles = [
        result_papers[i]["title"] for i in second_subgraph.edges()[1].cpu().numpy()
    ]
    for _src, _dst in zip(_src_titles, _dst_titles):
        links.append(
            {
                "source": _src,
                "target": _dst,
            }
        )

    d3_json = {
        "time": time.time() - start,
        "main_title": title_paper,
        "main_url": paper_url,
        "nodes": result_papers,
        "links": links,
        "is_existed": False,
    }
    print(">>> Pipeline: {}s".format(time.time() - start))
    gc.collect()

    return d3_json


@timer
def infer_existed_node(paper, topk):

    start = time.time()

    paper_id = paper["id"]
    in_paper_ids = paper["inCitations"]
    out_paper_ids = paper["outCitations"]
    abstract = paper["paperAbstract"]
    title_paper = paper["title"]
    paper_url = paper["s2Url"]

    # mapping to origin ss paper's id
    ss_origin_id = node_mapping[paper_id]
    ss_in_ids = [node_mapping[i] for i in in_paper_ids]
    ss_out_ids = [node_mapping[i] for i in out_paper_ids]

    # some papers have overlap cited paper in `inCitations` and `outCitations`
    ss_origin_id = node_mapping[paper_id]
    ss_in_ids = [node_mapping[i] for i in in_paper_ids]
    ss_out_ids = [node_mapping[i] for i in out_paper_ids]
    ss_inout_ids = set(ss_in_ids + ss_out_ids)
    ss_dgl_ids = list(ss_inout_ids.difference(set([ss_origin_id]))) + [ss_origin_id]
    assert ss_dgl_ids[-1] == ss_origin_id

    # ?? make k-hop subgraph, default 1-hop ??
    # ss_dgl_ids = make_khop_ids(g, ss_dgl_ids)

    # make 1st subgraph
    # no need to adding new edges
    first_subgraph = g.subgraph(torch.tensor(ss_dgl_ids))
    print(">>> 1st graph before adding new edges - {}s".format(time.time() - start))
    print(first_subgraph)

    id_subgraph2graph = {
        int(old_label): int(new_label)
        for new_label, old_label in zip(ss_dgl_ids, first_subgraph.nodes())
    }
    # id_graph2subgraph = dict(map(reversed, id_subgraph2graph.items()))

    # ?? node feature for query node
    # origin_node_feature = g.ndata["features"][node_mapping[paper_id]]
    # first_subgraph.ndata["features"][-1] = origin_node_feature

    # make inference on unseen node
    y_pred = model.inference(
        first_subgraph, first_subgraph.ndata["features"], batch_size, device
    )
    y_pred = y_pred.detach().numpy()

    no_nodes = first_subgraph.number_of_nodes()
    no_edges = first_subgraph.number_of_edges()
    degree = no_edges / no_nodes

    # TODO
    if degree < 30:
        topk = int(topk * 1.5)

    # get top-k most similarity
    dist_ids = np.linalg.norm(y_pred - y_pred[-1], axis=1)
    y_pred_ids = np.argsort(dist_ids)[1 : topk + 1]
    dist_ids = dist_ids[y_pred_ids]

    # make 2nd subgraph from top-k
    second_subgraph = first_subgraph.subgraph(
        torch.tensor(y_pred_ids.tolist() + [first_subgraph.number_of_nodes() - 1])
    )
    print(">>> 2nd graph - topk similarity - {}s".format(time.time() - start))
    print(second_subgraph)

    # TODO: add distance measure the similarity of each pair node

    paper_ids = [
        reversed_node_mapping[k]
        for k in [id_subgraph2graph[j] for j in [i for i in y_pred_ids]]
    ]
    result_papers = list(col_v4.find({"id": {"$in": paper_ids}}))
    result_papers.sort(key=lambda thing: paper_ids.index(thing["id"]))
    for index, paper in enumerate(result_papers):
        paper["dist"] = round(dist_ids[index], 4)
    result_papers.append(
        {"title": title_paper, "s2Url": paper_url, "paperAbstract": abstract}
    )

    _src_titles = []
    for i in second_subgraph.edges()[0].cpu().numpy():
        _src_titles.append(result_papers[i]["title"])

    _dst_titles = []
    for i in second_subgraph.edges()[1].cpu().numpy():
        _dst_titles.append(result_papers[i]["title"])

    # retrieve node for d3 visualize
    links = []
    for index, (_src, _dst) in enumerate(zip(_src_titles, _dst_titles)):
        links.append(
            {
                "source": _src,
                "target": _dst,
            }
        )

    d3_json = {
        "time": time.time() - start,
        "main_title": title_paper,
        "main_url": paper_url,
        "nodes": result_papers[:-1],
        "links": links,
        "is_existed": True,
    }

    print(">>> Pipeline: {}s".format(time.time() - start))
    gc.collect()

    return d3_json


@app.route("/")
def root():

    print(">" * 100)
    search_url = request.args.get("search")

    if search_url is not None:
        json_data = make_request_json(search_url)
        if json_data is None:
            sample_nodes = make_sample_nodes(sample_size=10)
            error_message = (
                "Invalid input url! Support Semantic Scholar, Arxiv, ACL url"
            )
            return render_template(
                "index.html", sample_nodes=sample_nodes, error_message=error_message
            )

        ss_paper_id = json_data["paperId"]

        return redirect(url_for("serve", ss_paper_id=ss_paper_id))

    else:
        sample_nodes = make_sample_nodes(sample_size=10)
        return render_template("index.html", sample_nodes=sample_nodes)


@app.route("/paper/<string:ss_paper_id>")
def serve(ss_paper_id):

    print(">" * 100)
    # check search bar is not empty
    search_url = request.args.get("search")
    error_message = None
    if search_url is not None:
        json_data = make_request_json(search_url)
        if json_data is None:
            error_message = (
                "Invalid input url! Support Semantic Scholar, Arxiv, ACL url"
            )
        else:
            ss_paper_id = json_data["paperId"]
            return redirect(url_for("serve", ss_paper_id=ss_paper_id))

    # check paper exists
    paper = list(col_v4.find({"id": ss_paper_id}))

    if len(paper) == 0:
        print(">>> Not exist")
        # unseen node
        # sample: 962dc29fdc3fbdc5930a10aba114050b82fe5a3e - detr
        ss_api_url = "https://api.semanticscholar.org/v1/paper/{}".format(ss_paper_id)
        header = {"User-Agent": str(ua.random)}
        req = requests.get(ss_api_url, headers=header)
        json_data = req.json()
        json_result = infer_unseen_node(json_data, topk)
    else:
        print(">>> Exist")
        # existed node
        # sample: 6b7d6e6416343b2a122f8416e69059ce919026ef - graphsage
        json_result = infer_existed_node(paper[0], topk)

    return render_template("index.html", **json_result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
