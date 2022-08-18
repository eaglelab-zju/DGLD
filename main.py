import sys
sys.path.append('./src')
from dgld.utils.evaluation import split_auc
from dgld.utils.common import seed_everything
from dgld.utils.argparser import parse_all_args
from dgld.utils.load_data import load_data
from dgld.utils.inject_anomalies import inject_contextual_anomalies,inject_structural_anomalies
from dgld.utils.common_params import Q_MAP,K,P
from dgld.models.DOMINANT import Dominant
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
import time
import os 
import json 

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == "__main__":
    args_dict,args = parse_all_args()
    seed_everything(args_dict['seed'])

    data_name = args_dict['dataset']
    save_path = args.save_path
    exp_name = args.exp_name
    if save_path is None:
        save_path = 'result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if exp_name is None:
        exp_name = args.model+'_'+data_name+'_'+str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    exp_path = save_path+'/'+exp_name
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    file_list = os.listdir(exp_path)
    id = 0
    for f in file_list:
        if f.startswith(exp_name):
            try:
                id_f = int(f.split('_')[-1])
                id = max(id_f+1,id)
            except:
                pass 
    exp_name += f'_{id}'
    exp_path += f'/{exp_name}'
    os.makedirs(exp_path)
    sys.stdout = Logger(exp_path+'/'+exp_name+'.log')
    
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
    print(args_dict)
    result = {}
    result['model'] = args.model
    result.update(vars(args))
    del result["save_path"]
    del result['exp_name']
    result['final anomaly score'] = final_score
    result['attribute anomaly score'] = a_score 
    result['structural anomaly score'] = s_score
    with open(exp_path+'/'+exp_name+'.json', 'w') as json_file:
        json_file.write(json.dumps(result, ensure_ascii=False, indent=4))
