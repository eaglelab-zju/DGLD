import sys
sys.path.append('./src')
import yaml 
from dgld.utils.evaluation import split_auc
from dgld.utils.common import seed_everything
from dgld.utils.log import Dgldlog
from dgld.utils.argparser import *
from dgld.data.dgldDataset  import NodeLevelAnomalyDataset
import random 
import os 
from dgld.models import *

if __name__ == "__main__":
    args = parse_all_args()
    if args.config_file is not None:
        with open(args.config_file,'r') as config_file:
            config = yaml.safe_load(config_file) 
            # update config 
            args_origin = vars(args)
            args_origin.update(config)
    # init log config 
    log = Dgldlog(args.save_path,args.exp_name,args)

    # result save 
    res_list_final = []
    res_list_attrb = []
    res_list_struct = []

    # rand seed list 
    seed_list = [random.randint(0,99999) for i in range(args.runs)]

    # if run times is one ,use fixed seed 
    if args.runs == 1:
        seed_list[0] = args.seed 

    for runs in range(args.runs):
        log.update_runs()

        args.seed = seed_list[runs] 
        seed_everything(args.seed) 

        graph = NodeLevelAnomalyDataset(args.dataset,args.category,args.data_path, **get_data_config(args))[0]

        # update graph info 
        args.feat_dim = graph.ndata['feat'].shape[1] 
        args.num_nodes = graph.num_nodes()

        # divide param 
        args_dict = param_divide(args) 

        # model init 
        try:
            model = eval(f'{args.model}(**args_dict["model"])')
        except:
            raise ValueError(f"{args.model} is not implemented!")
        
        # model fit 
        model.fit(graph, **args_dict["fit"])

        # model predict 
        result = model.predict(graph, **args_dict["predict"])

        # model evaluation
        label = graph.ndata['label'] 
        final_score, a_score, s_score = split_auc(label, result)
        res_list_final.append(final_score)
        res_list_attrb.append(a_score)
        res_list_struct.append(s_score)

    log.save_result(res_list_final,res_list_attrb,res_list_struct,seed_list,args)
    os._exit(0)   