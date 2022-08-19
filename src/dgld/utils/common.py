import numpy as np
import torch
import dgl
import json
from scipy.stats import rankdata
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.io as sio
import scipy.sparse as sp
import random,os
import os,wget,ssl,sys
import os.path as osp
import pandas as pd
from texttable import Texttable
from typing import *
current_file_name = __file__
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(current_file_name)))
data_path = current_dir +'/data/'

def print_shape(*a):
    for t in a:
        print(t.shape)

def print_format_dict(dict_input):
    """print dict with json for a decent show

    Parameters
    ----------
    dict_input : Dict
        dict to print
    """
    print(json.dumps(dict_input, indent=4, separators=(',', ':')))

def loadargs_from_json(filename, indent=4):
    """load args from a format json file

    Parameters
    ----------
    filename : file name
        json filename
    indent : int, optional
        description, by default 4

    Returns
    -------
    Dict : json
        args parameters
    """
    f = open(filename, "r") 
    content = f.read()
    args = json.loads(content)
    return args

def saveargs2json(jsonobject, filename, indent=4):
    """save args parameters to json with a decent format

    Parameters
    ----------
    jsonobject : Dict
        dict object to save
    filename : str
        file name
    indent : int, optional
        description, by default 4
    """
    with open(filename, "w") as write_file:
        json.dump(jsonobject, write_file, indent=indent, separators=(',', ':'))

def seed_everything(seed=42):
    # basic
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # dgl
    dgl.seed(seed)
    dgl.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

class ExpRecord():
    """create a read a existed csv file to record the experiments

    Parameters
    ----------
    filepath : str, filepath
        description, by default 'result.csv'
    
    Examples
    -------
    >>> exprecord = ExpRecord() 
    >>> argsdict = vars(args)
    >>> argsdict['auc'] = 1.0
    >>> argsdict['info'] = "test"
    >>> exprecord.add_record(argsdict)
    """ 
    def __init__(self, filepath='result.csv'): 
        self.filepath = filepath
        
    def add_record(self, dict_record):
        """summary

        Parameters
        ----------
        dict_record : Dict
            record to add 
        """
        
        if osp.exists(self.filepath):
            self.record = self.load_record()
        else:
            self.record = None
        print(dict_record)
        if not self.record:
            self.record = {k:[v] for k, v in dict_record.items()}
        else:
            for k in dict_record:
                # handle new column
                if k not in self.record:
                    self.record[k] = [''] * (len(self.record[list(self.record.keys())[0]])-1)
                self.record[k].append(dict_record[k])

            # check out parameters
            for k in self.record:
                if k not in dict_record:
                    self.record[k].append('')
            
        self.save_record()

    def save_record(self):
        pd.DataFrame(self.record).to_csv(self.filepath, index=None)
    
    def load_record(self):
        csv_file = pd.read_csv(self.filepath)
        self.record = {k:list(csv_file[k]) for k in csv_file.columns}
        return self.record 

class Multidict2dict():
    """convert multilayer Dict to a single layer Dict

    Parameters
    ----------
    inputs : Dict
    input Dict, Maybe multilayer like{
        {"a":{"as":"value"}}
    }

    Returns
    -------
    Dict: a single layer Dict

    Examples
    -------
    >>> tool = Multidict2dict()
    >>> inputs = {
    >>>    "1layer":{
    >>>        "2layer_one":{
    >>>            "3layers1":4,
    >>>            "3layers2":2,
    >>>        },
    >>>        "2layer_two":2
    >>>    }
    >>> }
    >>> result = tool.solve(inputs)
    >>> print(result)
    >>> {'3layers1': 4, '3layers2': 2, '2layer_two': 2}
    """
    def __init__(self):
        self.result = {}
    
    def solve(self, inputs):
        self.result = {}
        def helper(inputs):
            for k, v in inputs.items():
                if isinstance(v, Dict):
                    helper(v)
                else:
                    self.result[k] = v

        helper(inputs)
        return self.result


class ParameterShower():
    """show Parameter using texttable

    Examples
    -------
    >>> inputs = {
    >>>         "1layer":{
    >>>             "2layer_one":{
    >>>                 "3layers1":4,
    >>>                 "3layers2":2,
    >>>             },
    >>>             "2layer_two":2
    >>>         }
    >>>     }
    >>> 
    >>> tool = ParameterShower()
    >>> tool.show_multilayer(inputs)
    +------------+-------+
    |    Name    | Value |
    +============+=======+
    | 3layers1   | 4     |
    +------------+-------+
    | 3layers2   | 2     |
    +------------+-------+
    | 2layer_two | 2     |
    +------------+-------+
    """
    def __init__(self):
        self.tool = Multidict2dict()

    def show_dict(self, inputs: Dict) -> None:
        inputs_list = [("Name", "Value")] + [(k, v)for k, v in inputs.items()]
        table = Texttable()
        table.set_cols_align(["l", "l"])
        table.add_rows(inputs_list)
        print(table.draw() + "\n")

    def show_multilayer(self, inputs: Dict) -> None:
        inputs_simple = self.tool.solve(inputs)
        self.show_dict(inputs_simple)


def ranknorm(input_arr):
    """
    return the 1-norm of rankdata of input_arr

    Parameters
    ----------
    input_arr: list
        the data to be ranked

    Returns
    -------
    rank : numpy.ndarray
        the 1-norm of rankdata
    """
    return rankdata(input_arr, method='min') / len(input_arr)


def allclose(a, b, rtol=1e-4, atol=1e-4):
    """
    This function checks if a and b satisfy the condition:
    |a - b| <= atol + rtol * |b|

    Parameters
    ----------
    input : Tensor
        first tensor to compare
    other : Tensor
        second tensor to compare
    atol : float, optional
        absolute tolerance. Default: 1e-08
    rtol : float, optional
        relative tolerance. Default: 1e-05
    
    Returns
    -------
    res : bool
        True for close, False for not
    """
    return torch.allclose(a.float().cpu(),
                          b.float().cpu(), rtol=rtol, atol=atol)


def move_start_node_fisrt(pace, start_node):
    """
    return a new pace in which the start node is in the first place.

    Parameters
    ----------
    pace : list
        the subgraph of start node
    start_node: int
        target node

    Returns
    -------
    pace : list
        subgraph whose first value is start_node
    """
    if pace[0] == start_node: return pace
    for i in range(1, len(pace)):
        if pace[i] == start_node:
            pace[i] = pace[0]
            break
    pace[0] = start_node
    return pace


def is_bidirected(g):
    """
    Return whether the graph is a bidirected graph.
    A graph is bidirected if for any edge :math:`(u, v)` in :math:`G` with weight :math:`w`,
    there exists an edge :math:`(v, u)` in :math:`G` with the same weight.
    
    Parameters
    ----------
    g : DGL.graph
    
    Returns
    -------
    res : bool
        True for bidirected, False for not
    """
    src, dst = g.edges()
    num_nodes = g.num_nodes()

    # Sort first by src then dst
    idx_src_dst = src * num_nodes + dst
    perm_src_dst = torch.argsort(idx_src_dst, dim=0, descending=False)
    src1, dst1 = src[perm_src_dst], dst[perm_src_dst]

    # Sort first by dst then src
    idx_dst_src = dst * num_nodes + src
    perm_dst_src = torch.argsort(idx_dst_src, dim=0, descending=False)
    src2, dst2 = src[perm_dst_src], dst[perm_dst_src]

    return allclose(src1, dst2) and allclose(src2, dst1)


def load_mat_data2dgl(data_path, verbose=True):
    """
    load data from .mat file

    Parameters
    ----------
    data_path : str
        the file to read in
    verbose : bool, optional
        print info, by default True

    Returns
    -------
    graph : DGL.graph
        the graph read from data_path
    """
    mat_path = data_path
    data_mat = sio.loadmat(mat_path)
    adj = data_mat['Network']
    feat = data_mat['Attributes']
    # feat = preprocessing.normalize(feat, axis=0)
    truth = data_mat['Label']
    truth = truth.flatten()
    graph = dgl.from_scipy(adj)
    graph.ndata['feat'] = torch.from_numpy(feat.toarray()).to(torch.float32)
    graph.ndata['label'] = torch.from_numpy(truth).to(torch.float32)
    num_classes = len(np.unique(truth))

    if verbose:
        print()
        print('  DGL dataset')
        print('  NumNodes: {}'.format(graph.number_of_nodes()))
        print('  NumEdges: {}'.format(graph.number_of_edges()))
        print('  NumFeats: {}'.format(graph.ndata['feat'].shape[1]))
        print('  NumClasses: {}'.format(num_classes))
    if 'ACM' in data_path:
        print('ACM')
        return [graph]
    # # add reverse edges
    # srcs, dsts = graph.all_edges()
    # graph.add_edges(dsts, srcs)
    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    assert is_bidirected(graph) == True
    return [graph]


def load_ogbn_arxiv(raw_dir=data_path):
    """
    Read ogbn-arxiv from dgl.

    Parameters
    ----------
    raw_dir : str
        Data path. Supports user customization.

    returns
    -------
    graph : dgl.graph
        the graph of ogbn-arxiv
    """
    data = DglNodePropPredDataset(name="ogbn-arxiv", root=raw_dir)
    graph, _ = data[0]
    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    assert is_bidirected(graph) == True
    return [graph]


# create this bar_progress method which is invoked automatically from wget
def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def load_BlogCatalog(raw_dir=data_path):
    """
    load BlogCatalog dgl graph

    Parameters
    ----------
    raw_dir : str
        Data path. Supports user customization.
    
    Returns
    -------
    graph : DGL.graph
    Examples
    -------
    >>> graph=load_BlogCatalog()[0]
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    data_file = os.path.join(raw_dir, 'BlogCatalog.mat')
    if not os.path.exists(data_file):
        url = 'https://github.com/GRAND-Lab/CoLA/blob/main/raw_dataset/BlogCatalog/BlogCatalog.mat?raw=true'
        wget.download(url, out=data_file, bar=bar_progress)

    return load_mat_data2dgl(data_path=data_file)


def load_Flickr(raw_dir=data_path):
    """
    load Flickr dgl graph

    Parameters
    ----------
    raw_dir : str
        Data path. Supports user customization.
    
    Returns
    -------
    graph : DGL.graph

    Examples
    -------
    >>> graph=load_Flickr()[0]
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    data_file = os.path.join(raw_dir, 'Flickr.mat')
    if not os.path.exists(data_file):
        url = 'https://github.com/GRAND-Lab/CoLA/blob/main/raw_dataset/Flickr/Flickr.mat?raw=true'
        wget.download(url, out=data_file, bar=bar_progress)

    return load_mat_data2dgl(data_path=data_file)


def load_ACM(raw_dir=data_path):
    """load ACM dgl graph

    Parameters
    ----------
    raw_dir : str
        Data path. Supports user customization.
    
    Returns
    -------
    graph : DGL.graph

    Examples
    -------
    >>> graph=load_ACM()[0]
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    data_file = os.path.join(raw_dir, 'ACM.mat')
    if not os.path.exists(data_file):
        url = 'https://github.com/GRAND-Lab/CoLA/blob/main/dataset/ACM.mat?raw=true'
        wget.download(url, out=data_file, bar=bar_progress)

    return load_mat_data2dgl(data_path=data_file)

def cprint(x, color='green'):
    from termcolor import colored
    if color == 'info':
        color = 'green'
        x = 'INFO:' + str(x)
    if color == 'debug':
        color = 'red'
        x = 'DEBUG:' + str(x)
        
    print(colored(x, color))

def lcprint(*arr, color='green'):
    x = ':: '.join([str(i) for i in arr])
    from termcolor import colored
    if color.lower() == 'info':
        color = 'green'
        x = 'INFO:' + str(x)
    if color.lower() == 'debug':
        color = 'red'
        x = 'DEBUG:' + str(x)
        
    print(colored(x, color))


def tab_printer(args: Dict, thead: List[str] = None) -> None:
    """Function to print the logs in a nice tabular format.

    Args:
        args (Dict): Parameters used for the model.
    """
    args = vars(args) if hasattr(args, '__dict__') else args
    keys = sorted(args.keys())
    txt = Texttable()
    txt.set_precision(5)
    params = [["Parameter", "Value"] if thead is None else thead]
    params.extend([[
        k.replace("_", " "),
        f"{args[k]}" if isinstance(args[k], bool) else args[k]
    ] for k in keys])
    txt.add_rows(params)
    print(txt.draw())

def preprocess_features(features):
    """
    Functions that process features, here norm in row
    
    Parameters
    ----------
    features : torch.Tensor
        features to be processed
    
    Returns
    -------
    None
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.Tensor(features).float()