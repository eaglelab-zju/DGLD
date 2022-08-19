import sys
sys.path.append('./src')
from dgld.utils.evaluation import split_auc
from dgld.utils.common import seed_everything
from dgld.utils.argparser import parse_all_args
from dgld.utils.load_data import load_data
from dgld.utils.inject_anomalies import inject_contextual_anomalies,inject_structural_anomalies
from dgld.utils.common_params import Q_MAP,K,P
from dgld.utils.log import Dgldlog
from dgld.models.DOMINANT import DOMINANT
from dgld.models.AnomalyDAE import AnomalyDAE
from dgld.models.ComGA import ComGA
from dgld.models.DONE import DONE
from dgld.models.AdONE import AdONE
from dgld.models.CONAD import CONAD
from dgld.models.ALARM import ALARM
from dgld.models.ONE import ONE 
from dgld.models.GAAN import GAAN
from dgld.models.GUIDE import GUIDE
from dgld.models.CoLA import CoLA
from dgld.models.AAGNN import AAGNN_batch
from dgld.models.SLGAD import SLGAD
from dgld.models.ANEMONE import ANEMONE
from dgld.models.GCNAE import GCNAE
from dgld.models.MLPAE import MLPAE
from dgld.models.SCAN import SCAN
import random 
import numpy as np
import json 

if __name__ == "__main__":
    args_dict,args = parse_all_args()
    data_name = args_dict['dataset']
    save_path = args.save_path
    exp_name = args.exp_name
    log = Dgldlog(save_path,exp_name,args)
    exp_name,exp_path = log.get_path()
    seed_list = []
    res_list_final = []
    res_list_attrb = []
    res_list_struct = []
    for runs in range(args.runs):
        if args.runs > 1:
            log.update_runs()
        seed = args_dict['seed']
        while seed in seed_list:
            seed = random.randint(0,99999)
        seed_list.append(seed)
        seed_everything(seed)
        args_dict['seed'] = seed
        
        graph = load_data(data_name)

        graph = inject_contextual_anomalies(graph=graph,k=K,p=P,q=Q_MAP[data_name])
        graph = inject_structural_anomalies(graph=graph,p=P,q=Q_MAP[data_name])
        label = graph.ndata['label']

        if args.model in ['DOMINANT','AnomalyDAE','ComGA','DONE','AdONE','CONAD','ALARM','ONE','GAAN','GUIDE','CoLA',
                        'AAGNN', 'SLGAD','ANEMONE','GCNAE','MLPAE','SCAN']:
            model_name = args.model
            if model_name == 'AAGNN':
                model_name = 'AAGNN_batch'
            model = eval(f'{model_name}(**args_dict["model"])')
        else:
            raise ValueError(f"{args.model} is not implemented!")

        model.fit(graph, **args_dict["fit"])
        result = model.predict(graph, **args_dict["predict"])
        final_score, a_score, s_score = split_auc(label, result)
        res_list_final.append(final_score)
        res_list_attrb.append(a_score)
        res_list_struct.append(s_score)
        print(args_dict)

    # ----save result json----
    result = {}
    result['model'] = args.model
    result.update(vars(args))
    del result["save_path"]
    del result['exp_name']
    result['final anomaly score'] = np.mean(res_list_final)
    result['attribute anomaly score'] = np.mean(res_list_attrb)
    result['structural anomaly score'] = np.mean(res_list_struct)
    result['variance'] = np.std(res_list_final)
    with open(exp_path+'/'+exp_name+'.json', 'w') as json_file:
        json_file.write(json.dumps(result, ensure_ascii=False, indent=4))
    log.auc_result(res_list_final,res_list_attrb,res_list_struct,seed_list)
