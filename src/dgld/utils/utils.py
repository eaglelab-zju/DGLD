import numpy as np
import torch
import dgl
import random,os
import os.path as osp
import pandas as pd
from texttable import Texttable
from typing import *

def print_shape(*a):
    for t in a:
        print(t.shape)

def print_format_dict(dict_input):
    """[print dict with json for a decent show]
    Parameters
    ----------
    dict_input : [Dict]
        [dict to print]
    """
    print(json.dumps(dict_input, indent=4, separators=(',', ':')))

def loadargs_from_json(filename, indent=4):
    """[load args from a format json file]
    Parameters
    ----------
    filename : [file name]
        [json filename]
    indent : int, optional
        [description], by default 4

    Returns
    -------
    [Dict]
        [args parameters ]
    """
    f = open(filename, "r") 
    content = f.read()
    args = json.loads(content)
    return args

def saveargs2json(jsonobject, filename, indent=4):
    """[save args parameters to json with a decent format]
    Parameters
    ----------
    jsonobject : [Dict]
        [dict object to save]
    filename : [str]
        [file name]
    indent : int, optional
        [description], by default 4
    """
    with open(filename, "w") as write_file:
        json.dump(jsonobject, write_file, indent=indent, separators=(',', ':'))

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

class ExpRecord():
    def __init__(self, filepath='result.csv'):
        """[create a read a existed csv file to record the experiments]

        Parameters
        ----------
        filepath : str, filepath
            [description], by default 'result.csv'
        Examples
        --------
        ```python
        >>> exprecord = ExpRecord() 
        >>> argsdict = vars(args)
        >>> argsdict['auc'] = 1.0
        >>> argsdict['info'] = "test"
        >>> exprecord.add_record(argsdict)
        ```
        """
        self.filepath = filepath
        if osp.exists(self.filepath):
            self.record = self.load_record()
        else:
            self.record = None
    def add_record(self, dict_record):
        """[summary]

        Parameters
        ----------
        dict_record : [Dict]
            
        """
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
    """[convert multilayer Dict to a single layer Dict]
    Parameters
    ----------
    inputs : Dict
        [input Dict, Maybe multilayer like{
            {"a":{"as":"value"}}
        }]

    Returns
    -------
    Dict
        [a single layer Dict]
    Examples:
    ```python
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
    ```
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
    """[show Parameter using texttable]
    Examples:
    ---------
    ```python
    >>> inputs = {
    ...         "1layer":{
    ...             "2layer_one":{
    ...                 "3layers1":4,
    ...                 "3layers2":2,
    ...             },
    ...             "2layer_two":2
    ...         }
    ...     }
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
    ```
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

