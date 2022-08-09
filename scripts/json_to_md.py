import json
import argparse
import os 
def get_table_header(res_dict:dict):
    head = '<thead><tr>'
    head_second = '<tr>'
    height = 2
    head += f'<th rowspan="{height}" colspan="1" style="text-align:center">task</th>'
    for first_head in res_dict.keys():
        row_span = height
        col_span = 1 
        if isinstance(res_dict[first_head],dict):
            col_span = len(res_dict[first_head])
            row_span = 1
            for second_head in res_dict[first_head].keys():
                head_second += f'<th>{second_head}</th>'
        head += f'<th rowspan="{row_span}" colspan="{col_span}" style="text-align:center">{first_head}</th>'
    head_second+= '</tr>'
    head += '</tr>'
    head += head_second
    head += '</thead>'
    return head 
def add_table_value(res_dict:dict,id):
    body = f'<tr><td>task{id}</td>'
    for first_head in res_dict.keys():
        if isinstance(res_dict[first_head],dict):
            for second_head in res_dict[first_head].keys():
                body += f'<td>{res_dict[first_head][second_head]}</td>'

        else :
            body += f'<td>{res_dict[first_head]}</td>'
    body += '</tr>'
    return body 
def get_table(res_list,id_list):
    head = get_table_header(res_list[0])
    body = '<tbody'
    for id,res_dict in zip(id_list,res_list):
        body += add_table_value(res_dict,id)
    body += '</tbody>'
    table = '<table border="1">'+head+body+'</table>'
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
    args = parser.parse_args()   
    dir = args.save_path+'/'+args.exp_name
    dir_list = []
    id_list = []
    try:
        file_list = os.listdir(dir)
        for f in file_list:
            path = dir + '/' + f 
            if os.path.isdir(path) and f.startswith(args.exp_name):
                dir_list.append(path+'/'+f+'.json')
                id = int(f.split('_')[-1])
                id_list.append(id)
    except:
        raise('error')
    res_list = []
    for d in dir_list:
        with open(d) as f:
            res = json.load(f)
        res_list.append(res)
    html = get_table(res_list,id_list) 

    with open(f'{dir}'+f'/{args.exp_name}_summary.md','w') as f:
        f.write(html)