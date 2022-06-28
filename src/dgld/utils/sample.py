"""
This is a program about sample random walk.
"""
import os
import sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))) + '/utils/'
sys.path.append(current_dir)

import dgl
import torch
import numpy as np
from multiprocessing import Pool, Queue, Manager
import multiprocessing

from common import move_start_node_fisrt
from datetime import datetime


# TODO CoLA Sample
class BaseSubGraphSampling:
    """
    An abstract class for writing transforms on subgraph sampling.

    """

    def __call__(self, g, start_node):
        """
        functions to call class, undone, useless

        Parameters
        ----------
        g : DGL.Graph
            input graph to generative subgraph
        start_node : list
            input start nodes of random walk to generate subgraph

        Returns
        -------
        None
        """
        raise NotImplementedError

    def __repr__(self):
        """
        functions to get class name

        Parameters
        ----------
        None

        Returns
        -------
        out : string
            name of instance
        """
        return self.__class__.__name__ + "()"


class UniformNeighborSampling(BaseSubGraphSampling):
    """
    Uniform sampling Neighbors to generate subgraph.

    Parameters
    ----------
    length : int
        the size of subgraph (default 4)

    """

    def __init__(self, length=4):
        self.length = 4

    def __call__(self, g, start_nodes):
        """
        functions to call class to generate subgraph

        Parameters
        ----------
        g : DGL.Graph
            input graph to generative subgraph
        start_node : list
            input start nodes of random walk to generate subgraph

        Returns
        -------
        rwl : List[List]
            list of subgraph
        """
        rwl = []
        for node in start_nodes:
            pace = [node]
            successors = g.successors(node).numpy().tolist()
            # remove node and shuffle
            successors.remove(node)
            np.random.shuffle(successors)
            pace += successors[:self.length - 1]
            pace += [pace[0]] * max(0, self.length - len(pace))
            rwl.append(pace)
        return rwl


class CoLASubGraphSampling(BaseSubGraphSampling):
    """
    we adopt random walk with restart (RWR)
    as local subgraph sampling strategy due to its usability and efficiency.
    we fixed the size 𝑆 of the sampled subgraph (number of nodes in the subgraph) to 4.
    For isolated nodes or the nodes which belong to a community with a size smaller than
    the predetermined subgraph size, we sample the available nodes repeatedly until an
    overlapping subgraph with the set size is obtained."
    described in [CoLA Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113)
    
    Parameters
    ----------
    length : int
        size of subgraph

    Examples
    -------
    >>> cola_sampler = CoLASubGraphSampling()
    >>> g = dgl.graph(([0, 1, 2, 3, 6], [1, 2, 3, 4, 0]))
    >>> g = dgl.add_reverse_edges(g)
    >>> g = dgl.add_self_loop(g)
    >>> ans = cola_sampler(g, [1, 2, 3, 5])
    >>> print(ans)
    >>> [[1, 0, 2, 3], [2, 1, 0, 6], [3, 1, 2, 0], [5, 5, 5, 5]]
    """

    def __init__(self, length=4):
        self.length = 4

    def __call__(self, g, start_nodes):
        """
        add self_loop to handle isolated nodes as soon as
        the nodes which belong to a community with a size smaller than
        it is a little different from author's paper.
        
        Parameters
        ----------
        g: DGLGraph object
            input graph to generative subgraph
        start_nodes: a Tensor or array contain start node.
            input start nodes of random walk to generate subgraph
        
        Returns
        -------
        rwl: List[List]
            the list of subgraph generated
        """
        # newg = dgl.remove_self_loop(g)
        # newg = dgl.add_self_loop(newg)
        # length is Very influential to the effect of the model, maybe caused "remote" neighbor is
        # not "friendly" to Rebuild Properties.
        paces = dgl.sampling.random_walk(g, start_nodes, length=self.length * 3, restart_prob=0)[0]
        rwl = []
        for start, pace in zip(start_nodes, paces):
            pace = pace.unique().numpy()
            np.random.shuffle(pace)
            pace = pace[: self.length].tolist()
            pace = move_start_node_fisrt(pace, start)
            pace += [pace[0]] * max(0, self.length - len(pace))
            rwl.append(pace)
        return rwl


def generate_random_walk(g, start_nodes, length, multi_length, restart_prob, Q=None):
    """
    get random walk from block of target node by mutliThread accelerating and store in Queue if necessary
    
    Parameters
    ----------
    g: dgl.graph
        the graph to generate random walk
    start_nodes_block : list
        target node to generate random walk
    length : int
        the size of subgraph, default 4
    multi_length : int
        multitime of subgraph to get more node
    restart_prob : float
        probability of restart, which means return to target node after each hip
    Q : multiprocessing.Queue
        Queue to store random walk, default None

    Returns
    -------
    rwl : list[list]
        random walk from target nodes
    """
    prob_tensor = torch.full((length,), restart_prob)
    prob_tensor[0] = 0
    multi_times = multi_length * length
    # print(type(start_nodes))
    # start_nodes = [1]
    # print(type(start_nodes))
    # print(dgl.sampling.random_walk(g, start_nodes, length = length, restart_prob = prob_tensor)[0])
    # exit()
    if (len(start_nodes) == 1):
        paces = [dgl.sampling.random_walk(g, start_nodes, length=length, restart_prob=prob_tensor)[0][:, 1:] for _ in
                 range(multi_times)]
    else:
        paces = [dgl.sampling.random_walk(g, start_nodes, length=length, restart_prob=prob_tensor)[0][:, 1:] for _ in
                 range(multi_times)]
    paces = torch.cat(paces, dim=1)
    if Q != None:
        Q.put(paces[0])
    return paces
    # print(paces[0])
    # print(paces[0].shape)
    # exit()


def generate_random_walk_singleThread(g, start_nodes, length, multi_length, restart_prob, Q=None):
    """
    get random walk from block of target node by mutliThread accelerating and store in Queue if necessary
    
    Parameters
    ----------
    g: dgl.graph
        the graph to generate random walk
    start_nodes_block : list
        target node to generate random walk
    length : int
        the size of subgraph, default 4
    multi_length : int
        multitime of subgraph to get more node
    restart_prob : float
        probability of restart, which means return to target node after each hip
    Q : multiprocessing.Queue
        Queue to store random walk, default None

    Returns
    -------
    rwl : list[list]
        random walk from target nodes
    """
    prob_tensor = torch.full((length,), restart_prob)
    prob_tensor[0] = 0
    multi_times = multi_length * length
    paces = dgl.sampling.random_walk(g, start_nodes, length=length, restart_prob=prob_tensor)[0]
    # paces = [dgl.sampling.random_walk(g, start_nodes, length = length, restart_prob = prob_tensor)[0] for _ in range(multi_times)]
    paces = torch.cat(paces, dim=1)
    if Q != None:
        Q.put(paces[0])
    return paces
    # print(paces[0])
    # print(paces[0].shape)
    # exit()


def generate_random_walk_multiThread(g, start_nodes, length, multi_length, restart_prob):
    """
    get random walk from block of target node, by mutliThread accelerating
    
    Parameters
    ----------
    g: dgl.graph
        the graph to generate random walk
    start_nodes_block : list
        target node to generate random walk
    length : int
        the size of subgraph, default 4
    multi_length : int
        multitime of subgraph to get more node
    restart_prob : float
        probability of restart, which means return to target node after each hip

    Returns
    -------
    rwl : list[list]
        random walk from target nodes
    """
    cpu_number = multiprocessing.cpu_count()
    p = Pool(cpu_number)
    manager = multiprocessing.Manager()
    Q = manager.Queue()
    # Q = Queue()
    multi_times = multi_length * length
    rwl = []
    for i in range(multi_times):
        p.apply_async(generate_random_walk, args=(g, start_nodes, length, multi_length, restart_prob, Q,))
        rwl.append(Q.get())
    p.close()
    # p.join()
    rwl = torch.cat(rwl)
    return rwl


def generate_random_walk_multiThread_high_level(g, start_nodes_block, paces_block, length, multi_length, restart_prob,
                                                Q=None):
    """
    get random walk from block of target node by mutliThread accelerating, generate new random walk
    if the length of pace from target node is not enough, and store in Queue if necessary
    
    Parameters
    ----------
    g: dgl.graph
        the graph to generate random walk
    start_nodes_block : list
        target node to generate random walk
    paces_block : list[list]
        rough paces_block from target node
    length : int
        the size of subgraph, default 4
    multi_length : int
        multitime of subgraph to get more node
    restart_prob : float
        probability of restart, which means return to target node after each hip
    Q : multiprocessing.Queue
        Queue to store random walk, default None

    Returns
    -------
    rwl : list[list]
        random walk from target nodes
    """
    temp_paces = []
    # print(paces_block)
    for start, pace in zip(start_nodes_block, paces_block):
        # for i in range(len(start_nodes)):
        # start = start_nodes[i]
        # pace = paces[i]
        # print(start)
        # print(g.in_degrees(start))
        # exit()
        pace = pace.unique().numpy().tolist()

        # print(pace)
        # exit()
        if -1 in pace:
            pace.remove(-1)
        # if start in pace:
        #     pace.remove(start)
        # np.random.shuffle(pace)
        pace = pace[: length]
        # print(pace)
        # exit()
        # pace = move_start_node_fisrt(pace, start)
        retry_time = 0
        while len(pace) < length - 1:
            # pace = dgl.sampling.random_walk(g, start, length = self.length * 5, restart_prob = 0)[0]
            if False:
                restart_prob = 0.9
                multi_length = 5
                prob_tensor = torch.full((length,), restart_prob)
                prob_tensor[0] = 0
                multi_times = multi_length * length
                # pace = dgl.sampling.random_walk(g, start, length = self.length, restart_prob = prob_tensor)[0]
                pace = [dgl.sampling.random_walk(g, [start], length=length, restart_prob=prob_tensor)[0] for _ in
                        range(multi_times)]
                pace = torch.cat(pace, dim=1)
                # print(pace)
                # exit()
            # print('process_1')
            pace = generate_random_walk(g, [start], length=length, multi_length=5, restart_prob=0.9)
            pace = pace.tolist()[0]
            pace = list(filter((-1).__ne__, pace))
            # print(pace)
            pace = pace[:(length * 5)]
            pace = list({}.fromkeys(pace).keys())
            # print(pace)
            # exit()
            # print('process_2')

            # pace = generate_random_walk_multiThread(g, start, length = self.length, multi_length = 5, restart_prob = 0.9)
            # print(pace)
            # print(pace.shape)
            # exit()

            # pace = pace.unique().numpy().tolist()
            if -1 in pace:
                pace.remove(-1)
            # if start in pace:
            #     pace.remove(start)
            retry_time += 1
            if (len(pace) < length - 1) and (retry_time > 10):
                # print(type(pace))
                # print(pace)
                # exit()
                if pace == []:
                    pace = [start]
                pace = pace * (length)
        # pace = g.successors(start).tolist() * length
        pace.insert(0, start)
        pace = pace[:length]
        pace = move_start_node_fisrt(pace, start)
        # pace += [pace[0]] * max(0, self.length - len(pace))
        # print(pace)
        # exit()
        temp_paces.append(pace)

    if Q != None:
        # print(temp_paces)
        Q.put(temp_paces)
    return temp_paces
    # rwl.append(pace)


class SLGAD_SubGraphSampling(BaseSubGraphSampling):
    """
    we adopt random walk with restart (RWR)
    as local subgraph sampling strategy due to its usability and efficiency.
    we fixed the size 𝑆 of the sampled subgraph (number of nodes in the subgraph) to 4.
    For isolated nodes or the nodes which belong to a community with a size smaller than
    the predetermined subgraph size, we sample the available nodes repeatedly until an
    overlapping subgraph with the set size is obtained."
    described in [CoLA Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113)
    
    Parameters
    ----------
    length : int
        size of subgraph
    """

    def __init__(self, length=4):
        self.length = 4

    def __call__(self, g, start_nodes, block_ID = None):
        """
        add self_loop to handle isolated nodes as soon as
        the nodes which belong to a community with a size smaller than
        it is a little different from author's paper.
        
        Parameters
        ----------
        g: DGLGraph object
            input graph to generative subgraph
        start_nodes: a Tensor or array contain start node.
            input start nodes of random walk to generate subgraph
        
        Returns
        -------
        rwl: List[List]
            the list of subgraph generated
        """
        ori_g = g
        g = dgl.remove_self_loop(g)

        start_time = datetime.now()
        # newg = dgl.remove_self_loop(g)
        # newg = dgl.add_self_loop(newg)
        # length is Very influential to the effect of the model, maybe caused "remote" neighbor is
        # not "friendly" to Rebuild Properties.
        # paces = dgl.sampling.random_walk(g, start_nodes, length=self.length*3, restart_prob=0)[0]
        paces = generate_random_walk(g, start_nodes, length=self.length, multi_length=3, restart_prob=1.0)
        # print(paces[:20])
        # exit()
        end_time = datetime.now()
        # print('start time : ', start_time)
        # print('end time : ', end_time)
        # print('process time : ', end_time - start_time)
        start_time = datetime.now()
        # print(paces[:, 0: 5])
        # exit()
        rwl = []
        if block_ID == "full":
            block = int(multiprocessing.cpu_count())
        elif block_ID == None:
            block = max(1, int(multiprocessing.cpu_count() / 4))
        else:
            block = 1
        # block = 1
        total_len = len(start_nodes)
        bar = int(total_len / block) + 1
        # print(bar)
        start_nodes_block = [start_nodes[bar * i: bar * (i + 1)] for i in range(block)]
        paces_block = [paces[bar * i: bar * (i + 1)] for i in range(block)]
        # print(paces_block[0])
        # exit()
        p = Pool(block)
        manager = multiprocessing.Manager()
        # Q = manager.Queue()
        Q = None
        temp_results = []
        for i in range(block):
            temp_result = p.apply_async(generate_random_walk_multiThread_high_level,
                                        args=(g, start_nodes_block[i], paces_block[i], self.length, 5, 0.9, Q,))
            # print(Q.get())
            # rwl.extend(temp_result.get())
            temp_results.append(temp_result)
            # temp_result.wait()
            # print(i)
            # temp_result.wait()
            # print(temp_result.get())

        for i in temp_results:
            i.wait()  # 等待进程函数执行完毕
        for i in temp_results:
            if i.ready():  # 进程函数是否已经启动了
                if i.successful():  # 进程函数是否执行成功
                    # print(i.get())  # 进程函数返回值
                    rwl.extend(i.get())
        # print(rwl)
        p.close()
        # print(rwl)
        # print(len(rwl))
        # print(rwl)
        # exit()
        end_time = datetime.now()
        # print('start time : ', start_time)
        # print('end time : ', end_time)
        # print('process time : ', end_time - start_time)
        # exit()
        g = ori_g
        # print(rwl[:20])
        # exit()
        return rwl


if __name__ == "__main__":
    cola_sampler = CoLASubGraphSampling()
    g = dgl.graph(([0, 1, 2, 3, 6], [1, 2, 3, 4, 0]))
    g = dgl.add_reverse_edges(g)
    # g = dgl.add_self_loop(g) for isolated nodes
    ans = cola_sampler(g, [1, 2, 3, 5])
    print(ans)
    # [[1, 0, 2, 3], [2, 1, 0, 6], [3, 1, 2, 0], [5, 5, 5, 5]]
    subg = dgl.node_subgraph(g, ans[0])
    print(subg)
