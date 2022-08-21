import sys 
import os 
import time
import numpy as np
import pandas as pd 
import json

class Logger(object):
    def __init__(self, filename="Default.log",terminal = sys.stdout):
        self.terminal = terminal
        self.log = open(filename, "a")
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

class Dgldlog():
    def __init__(self,save_path,exp_name,args):
        if save_path is None:
            save_path = 'result'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if exp_name is None:
            exp_name = args.model+'_'+args.dataset+'_'+str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
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
        self.terminal = sys.stdout
        self.save_path = save_path
        self.exp_name = exp_name
        self.exp_path = exp_path 
        self.run_num = 0

    
    def update_runs(self):
        sys.stdout = Logger(self.exp_path+'/'+self.exp_name+f'_run{self.run_num}'+'.log',self.terminal)
        self.run_num += 1
    
    def auc_result(self,final_list,a_list,s_list,seed_list):
        index = []
        columns = ['final anomaly score','attribute anomaly score','structural anomaly score']
        for idx in range(len(final_list)):
            index.append('run'+str(idx))
        index.append('mean')
        index.append('variance')
        f_mean = np.mean(final_list)
        a_mean = np.mean(a_list)
        s_mean = np.mean(s_list)
        f_std = np.std(final_list)
        a_std = np.std(a_list)
        s_std = np.std(s_list)
        final_list.append(f_mean)
        a_list.append(a_mean)
        s_list.append(s_mean)
        seed_list.append('-')
        final_list.append(f_std)
        a_list.append(a_std)
        s_list.append(s_std)
        seed_list.append('-')
        res = pd.DataFrame({'seed':seed_list,
                            'final anomaly score':final_list,
                            'attribute anomaly score':a_list,
                            'structural anomaly score':s_list},index = index)
        res.to_markdown(self.exp_path+'/auc_res.md')

    def save_result(self,res_list_final,res_list_attrb,res_list_struct,seed_list,args):
            result = {}
            result['model'] = args.model
            result.update(vars(args))
            del result["save_path"]
            del result['exp_name']
            result['final anomaly score'] = np.mean(res_list_final)
            result['attribute anomaly score'] = np.mean(res_list_attrb)
            result['structural anomaly score'] = np.mean(res_list_struct)
            result['variance'] = np.std(res_list_final)
            with open(self.exp_path+'/'+self.exp_name+'.json', 'w') as json_file:
                json_file.write(json.dumps(result, ensure_ascii=False, indent=4))
            self.auc_result(res_list_final,res_list_attrb,res_list_struct,seed_list)