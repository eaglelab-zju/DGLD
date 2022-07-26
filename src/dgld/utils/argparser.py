"""Parse All Model Args"""
import sys
import sys
import os
import argparse
from typing import Dict
from common import tab_printer
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.abspath(current_file_name)))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
# DOMINANT
from models.DOMINANT import set_subargs as dominant_set_args
from models.DOMINANT import get_subargs as dominant_get_args
#AnomalyDAE
from models.AnomalyDAE import set_subargs as anomalydae_set_args
from models.AnomalyDAE import get_subargs as anomalydae_get_args
# ComGA
from models.ComGA import set_subargs as comga_set_args
from models.ComGA import get_subargs as comga_get_args
# DONE
from models.DONE import set_subargs as done_set_args
from models.DONE import get_subargs as done_get_args
# AdONE
from models.AdONE import set_subargs as adone_set_args
from models.AdONE import get_subargs as adone_get_args
# CONAD
from models.CONAD import set_subargs as conad_set_args
from models.CONAD import get_subargs as conad_get_args


# set args 
models_set_args_map = {
    "DOMINANT": dominant_set_args,
    "AnomalyDAE": anomalydae_set_args,
    "ComGA": comga_set_args,
    "DONE": done_set_args,
    "AdONE": adone_set_args,
    "CONAD": conad_set_args,
}
# get args 
models_get_args_map = {
    "DOMINANT": dominant_get_args,
    "AnomalyDAE": anomalydae_get_args,
    "ComGA": comga_get_args,
    "DONE": done_get_args,
    "AdONE": adone_get_args,
    "CONAD": conad_get_args,
}


def parse_all_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='DGLD',
        description='Parameters for DGLD')
    parser.add_argument('--dataset',
                        type=str,
                        default='Cora',
                        help='Dataset used in the experiment')
    parser.add_argument('--device',
                        type=str,
                        default='0',
                        help='ID(s) of gpu used by cuda')
    parser.add_argument('--seed',
                        type=int,
                        default=4096,
                        help='Random seed. Defaults to 4096.')

    subparsers = parser.add_subparsers(dest="model", help='sub-command help')
    
    # set sub args
    for _model, set_arg_func in models_set_args_map.items():
        sub_parser = subparsers.add_parser(
            _model, help=f"Run anomaly detection on {_model}")
        set_arg_func(sub_parser)
        
    # get model args
    args = parser.parse_args()
    args_dict,args = models_get_args_map[args.model](args)

    tab_printer(args)
    
    return args_dict,args

if __name__ == "__main__":
    parse_all_args() 