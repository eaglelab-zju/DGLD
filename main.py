import sys
sys.path.append('./src')

from dgld.utils.dataset import GraphNodeAnomalyDectionDataset
from dgld.utils.evaluation import split_auc
from dgld.utils.common import seed_everything
from dgld.utils.argparser import parse_all_args
from dgld.utils.load_data import load_data

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

if __name__ == "__main__":
    args_dict,args = parse_all_args()
    seed_everything(args_dict['seed'])
    gnd_dataset = GraphNodeAnomalyDectionDataset(args_dict['dataset'])
    g = gnd_dataset[0]
    label = gnd_dataset.anomaly_label

    if args.model == 'DOMINANT':
        model = Dominant(**args_dict["model"])
    elif args.model == 'AnomalyDAE':
        model = AnomalyDAE(**args_dict["model"])
    elif args.model == 'ComGA':
        model = ComGA(**args_dict["model"])
    elif args.model == 'DONE':
        model = DONE(**args_dict["model"])
    elif args.model == 'AdONE':
        model = AdONE(**args_dict["model"])
    elif args.model == 'CONAD':
        model = CONAD(**args_dict["model"])
    elif args.model == 'ALARM':
        model = ALARM(**args_dict["model"])
    elif args.model == 'ONE':
        model = ONE(**args_dict["model"])
    elif args.model == 'GAAN':
        model = GAAN(**args_dict["model"])
    elif args.model == 'GUIDE':
        model = GUIDE(**args_dict["model"])
    elif args.model == 'CoLA':
        model = CoLA(**args_dict["model"])   

    else:
        raise ValueError(f"{args.model} is not implemented!")

    model.fit(g, **args_dict["fit"])
    result = model.predict(g, **args_dict["predict"])
    split_auc(label, result)
    print(args_dict)