import numpy as np 
import os,sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
if current_dir not in sys.path:
    sys.path.append(current_dir)


def set_subargs(parser):
    parser.add_argument('--num_epoch', type=int, default=5, help='Training epoch')
    parser.add_argument('--K', type=int, default=8,help='The embedding size of graph nodes')
    parser.add_argument('--alpha', type=float, default=1,
                        help='balance parameter')
    parser.add_argument('--beta', type=float, default=1,
                        help='balance parameter')
    parser.add_argument('--gamma', type=float, default=1,
                        help='balance parameter')
def get_subargs(args):
    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model":{
            "node_num":args.num_nodes,
            "K":args.K
        },
        "fit":{
            "num_epoch":args.num_epoch,
        },
        "predict":{
            "alpha":args.alpha,
            "beta":args.beta,
            "gamma":args.gamma,
        }
    }
    return final_args_dict,args


def loss_func(A, C, G, H, U, V, W, outl1, outl2, outl3, alpha, beta, gamma):
    eps = 1e-5
    temp1 = A - np.matmul(G,H)
    temp1 = np.multiply(temp1,temp1)
    temp1 = np.multiply( np.log(np.reciprocal(outl1+eps)), np.sum(temp1, axis=1) )
    temp1 = np.sum(temp1)
                        
    temp2 = C - np.matmul(U,V)
    temp2 = np.multiply(temp2,temp2)
    temp2 = np.multiply( np.log(np.reciprocal(outl2+eps)), np.sum(temp2, axis=1) )
    temp2 = np.sum(temp2)    
    
    temp3 = G.T - np.matmul(W, U.T)
    temp3 = np.multiply(temp3,temp3)
    temp3 = np.multiply( np.log(np.reciprocal(outl3+eps)), np.sum(temp3, axis=0).T )
    temp3 = np.sum(temp3)
    
    func_value = alpha * temp1 + beta * temp2 + gamma * temp3
    return func_value