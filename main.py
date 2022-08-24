import sys
sys.path.append('./src')
from dgld.utils.evaluation import split_auc
from dgld.utils.common import seed_everything
from dgld.utils.argparser import parse_all_args
from dgld.utils.load_data import load_data,load_custom_data, load_truth_data
from dgld.utils.inject_anomalies import inject_contextual_anomalies,inject_structural_anomalies
from dgld.utils.common_params import Q_MAP,K,P
from dgld.utils.log import Dgldlog
from dgld.models import *
import random 
import os 
truth_list = ['weibo','tfinance','tsocial','reddit','Amazon','Class','Disney','elliptic','Enron']
if __name__ == "__main__":
    args_dict,args = parse_all_args()
    data_name = args_dict['dataset']
    save_path = args.save_path
    exp_name = args.exp_name
    log = Dgldlog(save_path,exp_name,args)

    res_list_final = []
    res_list_attrb = []
    res_list_struct = []

    seed_list = [random.randint(0,99999) for i in range(args.runs)]
    seed_list[0] = args_dict['seed']
    for runs in range(args.runs):
        log.update_runs()
        seed = seed_list[runs]
        seed_everything(seed)
        args_dict['seed'] = seed
        
        if data_name in truth_list:
            graph = load_truth_data(data_path=args.data_path,dataset_name=data_name)
        elif data_name == 'custom':
            graph = load_custom_data(data_path=args.data_path)
        else:
            graph = load_data(data_name)
            graph = inject_contextual_anomalies(graph=graph,k=K,p=P,q=Q_MAP[data_name],seed=seed)
            graph = inject_structural_anomalies(graph=graph,p=P,q=Q_MAP[data_name],seed=seed)

        label = graph.ndata['label']

        if args.model in ['DOMINANT','AnomalyDAE','ComGA','DONE','AdONE','CONAD','ALARM','ONE','GAAN','GUIDE','CoLA',
                        'AAGNN', 'SLGAD','ANEMONE','GCNAE','MLPAE','SCAN']:
            model = eval(f'{args.model}(**args_dict["model"])')
        else:
            raise ValueError(f"{args.model} is not implemented!")

        model.fit(graph, **args_dict["fit"])
        result = model.predict(graph, **args_dict["predict"])
        final_score, a_score, s_score = split_auc(label, result)
        res_list_final.append(final_score)
        res_list_attrb.append(a_score)
        res_list_struct.append(s_score)
        print(args_dict)

    log.save_result(res_list_final,res_list_attrb,res_list_struct,seed_list,args)
    os._exit(0)
