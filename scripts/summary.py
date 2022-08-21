import json
import argparse
import os 
import warnings 
import pandas as pd 
def get_table_header(res_dict:dict):
    head = ['task']
    for first_head in res_dict.keys():
        head.append(first_head)
    return head 

def add_table_value(res_dict:dict,id):
    body = [f'task{id}']
    for first_head in res_dict.keys():
        body.append(res_dict[first_head])
    return body 

def get_table(res_list,id_list):
    head = get_table_header(res_list[0])
    body = []
    model_name = None 
    same_flag = True
    for id,res_dict in zip(id_list,res_list):
        data = add_table_value(res_dict,id)
        body.append(data)
        if model_name is None:
            model_name = data[1]
        else:
            if model_name != data[1]:
                same_flag = False
    if not same_flag:
        warnings.warn('There are different models in this experiment, and the parameters cannot be aligned. Only the results are summarized.')
        for i,data in enumerate(body):
            data = data[:3] + data[-4:]
            body[i] = data 
        head = ["task","model","dataset","final anomaly score","attribute anomaly score","structural anomaly score","variance"]
    table = pd.DataFrame(body,columns=head)
    return table,same_flag

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='DGLD',
        description='Summary result json of one experiment to one table')
    parser.add_argument('--save_path',
                        type=str,
                        default='result',
                        help='save path of the result')
    parser.add_argument('--exp_name',
                        type=str,
                        help='exp_name experiment identification')
    parser.add_argument('--filter',
                        action='store_true',
                        help='filter same parameter in summary table')
    args = parser.parse_args()   
    if args.exp_name is None :
        raise NameError('exp_name is none')
    dir = args.save_path+'/'+args.exp_name
    dir_list = []
    id_list = []
    file_list = os.listdir(dir)
    for f in file_list:
        path = dir + '/' + f 
        try:
            if os.path.isdir(path) and f.startswith(args.exp_name) and os.path.exists(path+'/'+f+'.json'):
                id = int(f.split('_')[-1])
                dir_list.append(path+'/'+f+'.json')
                id_list.append(id)
        except:
            pass 

    res_list = []
    for d in dir_list:
        with open(d) as f:
            res = json.load(f)
        res_list.append(res)
    table,same_flag = get_table(res_list,id_list) 
    if args.filter :
        if table.shape[0] > 1:
            col = list(table.columns)
            finally_col = col[:3]
            for c in col[3:-4]:
                li = table[c].astype(str).values
                s_li = set(li)
                if len(s_li) != 1:
                    finally_col.append(c)
            finally_col += col[-4:]
            table = table[finally_col]
    table['sort_value'] = table['task'].apply(lambda x : int(x[4:]))
    table = table.sort_values(by='sort_value')
    del table['sort_value']
    table.to_markdown(f'{dir}'+f'/{args.exp_name}_summary.md',index=False)
    if not same_flag:
        with open(f'{dir}'+f'/{args.exp_name}_summary.md','a+') as f:
            f.write('\n')
            f.write('- There are different models in this experiment, and the parameters cannot be aligned. Only the results are summarized.')