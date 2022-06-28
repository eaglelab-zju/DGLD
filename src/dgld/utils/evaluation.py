"""
This is a program about evaluating scores.
"""

from sklearn.metrics import roc_auc_score
import numpy as np

def split_auc(groundtruth, prob):
    """
    print the scoring(AUC) of the two types of anomalies separately and global auc.

    Parameters
    ----------
    groundtruth: np.ndarray
        Indicates whether this node is an injected anomaly node.
            0: normal node
            1: structural anomaly
            2: contextual anomaly

    prob: np.ndarray-like array
        saving the predicted score for every node
    
    Returns
    -------
    None
    """
    s_score = -1
    a_score = -1
    try:
        str_pos_idx = groundtruth == 1
        attr_pos_idx = groundtruth == 2
        norm_idx = groundtruth == 0

        str_data_idx = str_pos_idx | norm_idx
        attr_data_idx = attr_pos_idx | norm_idx

        str_data_groundtruth = groundtruth[str_data_idx]
        str_data_predict = prob[str_data_idx]

        attr_data_groundtruth = np.where(groundtruth[attr_data_idx] != 0, 1, 0)
        attr_data_predict = prob[attr_data_idx]

        s_score = roc_auc_score(str_data_groundtruth, str_data_predict)
        a_score = roc_auc_score(attr_data_groundtruth, attr_data_predict)
        print("structural anomaly score:", s_score)
        print("attribute anomaly score:", a_score)
    except ValueError:
        pass
    final_score = roc_auc_score(np.where(groundtruth == 0, 0, 1), prob)

    print("final anomaly score:", final_score)
    return final_score, a_score, s_score