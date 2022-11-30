import numpy as np


def cal_hamming_dis(b1, b2):
    k = b2.shape[1]  # length of hash code
    dis = 0.5 * (k - np.dot(b1, b2.transpose()))
    return dis


def cal_cosine_dis(f1, f2):
    f1_norm = np.linalg.norm(f1)
    f2_norm = np.linalg.norm(f2, axis=1)

    similiarity = np.dot(f1, f2.T)/(f1_norm * f2_norm) 
    return 1 - similiarity


def cal_map(query_feats, query_label, retrieval_feats, retrieval_label, top_k=5000, dist_method='hamming'):
    """
    Calculate MAP (Mean Average Precision)
    :param query_binary: binary code of query sample
    :param query_label: label of qurey sample
    :param retrieval_binary: binary code of database
    :param retrieval_label: label of database
    :param top_k:
    :return:
    """
    query_number = query_label.shape[0]
    top_k_map = 0

    dist_func = cal_hamming_dis if dist_method == 'hamming' else cal_cosine_dis

    for query_index in range(query_number):
        # (1, N)
        ground_truth = (np.dot(query_label[query_index, :], retrieval_label.transpose()) > 0).astype(np.float32)
        hamming_dis = dist_func(query_feats[query_index, :], retrieval_feats)  # (1, N)

        # sort hamming distance
        sort_index = np.argsort(hamming_dis)

        # resort ground truth
        ground_truth = ground_truth[sort_index]

        # get top K ground truth
        top_k_gnd = ground_truth[0:top_k]
        top_k_sum = np.sum(top_k_gnd).astype(int)  # the number of correct retrieval in top K
        if top_k_sum == 0:
            continue
        count = np.linspace(1, top_k_sum, int(top_k_sum))

        top_k_index = np.asarray(np.where(top_k_gnd == 1)) + 1.0
        top_k_map += np.mean(count / top_k_index)  # average precision of per class

    return top_k_map / query_number  # mean of average precision of all class


def cal_pr(retrieval_binary, query_binary, retrieval_label, query_label, interval=0.1):
    r_arr = np.array([i * interval for i in range(1, int(1/interval) + 1)])
    p_arr = np.zeros(len(r_arr))

    query_number = query_label.shape[0]

    for query_index in range(query_number):
        ground_truth = (np.dot(query_label[query_index, :], retrieval_label.transpose()) > 0).astype(
            np.float32)  # (1, N)
        hamming_dis = cal_hamming_dis(query_binary[query_index, :], retrieval_binary)  # (1, N)

        # sort hamming distance
        sort_index = np.argsort(hamming_dis)
        ground_truth = ground_truth[sort_index]
        tp_num = len(np.where(ground_truth == 1)[0])
        r_num_arr = (tp_num * r_arr).astype(np.int32)

        tp_cum = np.cumsum(ground_truth)
        total_num_arr = np.array([np.where(tp_cum == i)[0][0] + 1 for i in r_num_arr])
        p_arr += r_num_arr/total_num_arr
    p_arr /= query_number

    return np.array(list(zip(r_arr, p_arr)))


def cal_top_n(retrieval_binary, query_binary, retrieval_label, query_label, top_n=None):
    if top_n is None:
        top_n = range(100, 1001, 100)

    top_n = np.array(top_n)
    top_n_p = np.zeros(len(top_n))
    query_number = query_label.shape[0]

    for query_index in range(query_number):
        ground_truth = (np.dot(query_label[query_index, :], retrieval_label.transpose()) > 0).astype(
            np.float32)  # (1, N)
        hamming_dis = cal_hamming_dis(query_binary[query_index, :], retrieval_binary)  # (1, N)

        # sort hamming distance
        sort_index = np.argsort(hamming_dis)
        ground_truth = ground_truth[sort_index]
        ground_truth = ground_truth[:top_n[-1]]

        tp_cum = np.cumsum(ground_truth)
        tp_num_arr = tp_cum[top_n - 1]
        top_n_p += tp_num_arr/top_n

    top_n_p /= query_number
    return np.array(list(zip(top_n, top_n_p)))
