import json
import argparse
import os 
import pandas as pd 
def get_table_header(res_dict:dict):
    head = ['task']
    for first_head in res_dict.keys():
        if isinstance(res_dict[first_head],dict):
            for second_head in res_dict[first_head].keys():
                head.append(first_head+'.'+second_head)
        else :
            head.append(first_head)
    return head 
def add_table_value(res_dict:dict,id):
    body = [f'task{id}']
    for first_head in res_dict.keys():
        if isinstance(res_dict[first_head],dict):
            for second_head in res_dict[first_head].keys():
                body.append(res_dict[first_head][second_head])
        else :
            body.append(res_dict[first_head])
    return body 
def get_table(res_list,id_list):
    head = get_table_header(res_list[0])
    body = []
    for id,res_dict in zip(id_list,res_list):
        body.append(add_table_value(res_dict,id))
    table = pd.DataFrame(body,columns=head)
    return table

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
    print(args)
    dir = args.save_path+'/'+args.exp_name
    dir_list = []
    id_list = []
    file_list = os.listdir(dir)
    for f in file_list:
        path = dir + '/' + f 
        try:
            if os.path.isdir(path) and f.startswith(args.exp_name):
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
    table = get_table(res_list,id_list) 
    print(args.filter)
    if args.filter :
        print('filter')
        col = list(table.columns)
        finally_col = col[:3]
        for c in col[3:-3]:
            li = table[c].values
            flag = False
            for i in range(1,len(li)):
                if str(li[i]) != str(li[i-1]):
                    flag = True
                    break
            if flag:
                finally_col.append(c)
        finally_col += col[-3:]
        table = table[finally_col]
    table.to_markdown(f'{dir}'+f'/{args.exp_name}_summary_v2.md',index=False)

