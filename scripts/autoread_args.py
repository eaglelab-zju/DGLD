import os
import sys
sys.path.append('./src')
from dgld.utils.argparser import models_set_args_map 
from dgld.utils.common import saveargs2json, loadargs_from_json
from copy import deepcopy

class FakeParser():
    def __init__(self):
        self.options = []
        
    def add_argument(self, option, **kwargs):
        if not option.startswith("--"):
            raise ValueError(f"{option} is not long option!!!")
        
        def get_type(kwargs):
            if "nargs" in kwargs.keys():
                return "list"
            elif "type" in kwargs.keys():
                return "bool" if kwargs['type'].__name__ == "<lambda>" else kwargs['type'].__name__
            else:
                return None
            
        def get_choices(kwargs):
            if get_type(kwargs) == "bool":
                return [True, False]
            else:
                return kwargs.get("choices", None)
                
        self.options.append({
            "name": option.lstrip('-'),
            "tip": kwargs.get("help", ""),
            "type": get_type(kwargs),
            "default": kwargs.get("default", None),
            "choices": get_choices(kwargs)
        })
        
        
if __name__ == "__main__":
    save_path = "hyper/"
    dataset_list = ['Cora', 'Citeseer', 'Pubmed', 'BlogCatalog', 'Flickr', 'ACM', 'ogbn-arxiv']
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    for _model, set_args in models_set_args_map.items():
        fp = FakeParser()
        set_args(fp)
        
        for dataset in dataset_list:
            options = deepcopy(fp.options)
            filepath = f'src/dgld/config/{_model}.json'
            config = {}
            
            if os.path.exists(filepath):
                config = loadargs_from_json(filepath).get(dataset, {})

            for i, x in enumerate(options):
                if x['name'] in config.keys():
                    options[i]['default'] = config[x['name']]
                    
            hyper = {"hyper": options}
            saveargs2json(hyper, save_path+f"{_model}_{dataset}.json")
        
        
    