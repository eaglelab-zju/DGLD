"""Parse All Model Args"""
import sys
import sys
import os
import argparse
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
# CONAD
from models.CONAD import set_subargs as conad_set_args
from models.CONAD import get_subargs as conad_get_args
# ALARM
from models.ALARM import set_subargs as alarm_set_subargs
from models.ALARM import get_subargs as alarm_get_subargs
# ONE
from models.ONE import set_subargs as one_set_args 
from models.ONE import get_subargs as one_get_args
# GAAN
from models.GAAN import set_subargs as gaan_set_args
from models.GAAN import get_subargs as gaan_get_args
# GUIDE
from models.GUIDE import set_subargs as guide_set_args
from models.GUIDE import get_subargs as guide_get_args
# CoLA
from models.CoLA import set_subargs as cola_set_args
from models.CoLA import get_subargs as cola_get_args
# SL-GAD
from models.SLGAD import set_subargs as slgad_set_args
from models.SLGAD import get_subargs as slgad_get_args
# AAGNN
from models.AAGNN import set_subargs as aagnn_set_args
from models.AAGNN import get_subargs as aagnn_get_args

# set args
models_set_args_map = {
    "DOMINANT": dominant_set_args,
    "AnomalyDAE": anomalydae_set_args,
    "ComGA": comga_set_args,
    "DONE": done_set_args,
    "CONAD": conad_set_args,
    "ALARM": alarm_set_subargs,
    "ONE": one_set_args,
    "GAAN": gaan_set_args,
    "GUIDE": guide_set_args,
    "CoLA": cola_set_args,
    "SLGAD": slgad_set_args,
    "AAGNN": aagnn_set_args
}
# get args
models_get_args_map = {
    "DOMINANT": dominant_get_args,
    "AnomalyDAE": anomalydae_get_args,
    "ComGA": comga_get_args,
    "DONE": done_get_args,
    "CONAD": conad_get_args,
    "ALARM": alarm_get_subargs,
    "ONE": one_get_args,
    "GAAN": gaan_get_args,
    "GUIDE":guide_get_args,
    "CoLA": cola_get_args,
    "SLGAD":slgad_get_args,
    "AAGNN": aagnn_get_args
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
