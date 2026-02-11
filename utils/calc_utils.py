import numpy as np
import torch
from sklearn.preprocessing import normalize


def calc_neighbor(a: torch.Tensor, b: torch.Tensor):
    return (a.matmul(b.transpose(0, 1)) > 0).float()


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_map_k_matrix(qB, rB, query_L, retrieval_L, k=None, rank=0):
    
    num_query = query_L.shape[0]
    if qB.is_cuda:
        qB = qB.cpu()
        rB = rB.cpu()
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        gnd = (query_L[iter].unsqueeze(0).mm(retrieval_L.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_recall_at_k(qB: torch.Tensor, rB: torch.Tensor,
                     qL: torch.Tensor, rL: torch.Tensor,
                     Ks: list = [1, 10, 100, 1000]) -> dict:
    """
    计算多标签场景下的Recall@K
    :param qB: 查询集哈希码 (n_query, n_bits)，值域为±1
    :param rB: 数据库集哈希码 (n_retrieval, n_bits)，值域为±1
    :param qL: 查询集标签 (n_query, n_classes)，多标签独热编码
    :param rL: 数据库集标签 (n_retrieval, n_classes)，多标签独热编码
    :param Ks: K值列表，如[1,10,100,1000]
    :return: 各K值对应的Recall@K字典
    """
    if qB.is_cuda:
        qB = qB.cpu()
        rB = rB.cpu()
    n_query = qB.shape[0]
    max_k = max(Ks)

    # 二值化哈希码到0/1 (提升汉明距离计算效率)
    qB = (qB > 0).type(torch.bool)  # True=1, False=0
    rB = (rB > 0).type(torch.bool)

    # 计算汉明距离矩阵 (利用矩阵运算加速)
    # (n_query, n_retrieval)
    hamm_dist = (qB.unsqueeze(1) != rB.unsqueeze(0)).sum(dim=2)

    # 按距离排序获取top-k索引
    _, topk_indices = torch.topk(hamm_dist, k=max_k, dim=1, largest=False, sorted=True)

    # 计算相关性矩阵 (多标签场景: 至少有一个共同标签)
    # (n_query, n_retrieval)
    relevance = (qL @ rL.t() > 0).type(torch.float)

    # 初始化用于保存Recall@K的字典
    recall_dict = {k: 0.0 for k in Ks}
    valid_queries = 0  # 有相关样本的查询数

    for i in range(n_query):
        # 获取当前查询的相关样本总数
        n_relevant = relevance[i].sum().item()
        if n_relevant == 0:
            continue  # 跳过无相关样本的查询
        valid_queries += 1

        # 获取前max_k个结果的标签
        topk_rel = relevance[i, topk_indices[i]]

        # 对每个K值计算Recall@K
        for k in Ks:
            # 获取前k个结果中的相关样本数
            n_relevant_topk = topk_rel[:k].sum().item()
            # 计算Recall@K：相关样本数除以K
            recall_at_k = n_relevant_topk / k
            recall_dict[k] += recall_at_k

    # 计算所有有效查询的平均召回率
    if valid_queries > 0:
        recall_dict = {k: recall_dict[k] / valid_queries for k in Ks}
    return recall_dict


def calc_ndcg_at_k_matrix(qF, rF, qL, rL, what=1, k=-1):
    """Normalized Discounted Cumulative Gain
    ref: https://github.com/kunhe/TALR/blob/master/%2Beval/NDCG.m
    """

    n_query = qF.shape[0]
    if qF.is_cuda:
        qF = qF.cpu()
        rF = rF.cpu()
    if (k < 0) or (k > rF.shape[0]):
        k = rF.shape[0]
    Rel = np.dot(qL, rL.T)
    G = 2 ** Rel - 1
    D = np.log2(2 + np.arange(k))
    if what == 0:
        Rank = np.argsort(1 - cos(qF, rF))
    elif what == 1:
        Rank = np.argsort(calc_hammingDist(qF, rF))
    elif what == 2:
        Rank = np.argsort(euclidean(qF, rF))

    _NDCG = 0
    for g, rnk in zip(G, Rank):
        dcg_best = (np.sort(g)[::-1][:k] / D).sum()
        if dcg_best > 0:
            dcg = (g[rnk[:k]] / D).sum()
            _NDCG += dcg / dcg_best
    return _NDCG / n_query


def calc_crc_k_matrix(qB_img, qB_txt, rB_img, rB_txt, L_query, L_db, K=100):
    num_query = L_query.shape[0]
    if qB_img.is_cuda:
        qB_img, qB_txt, rB_img, rB_txt = qB_img.cpu(), qB_txt.cpu(), rB_img.cpu(), rB_txt.cpu()

    crc = 0

    for i in range(num_query):
        # 计算 i→t（图像查询文本）的排名
        gnd_it = (L_query[i].unsqueeze(0).mm(L_db.t()) > 0).type(torch.float).squeeze()
        if torch.sum(gnd_it) == 0:
            continue
        hamm_it = calc_hammingDist(qB_img[i, :], rB_txt)
        _, ind_it = torch.sort(hamm_it)
        ind_it.squeeze_()
        gnd_it = gnd_it[ind_it][:K]  # 只考虑前 K 个检索结果
        rank_it = (torch.nonzero(gnd_it)[0] + 1).item() if torch.sum(gnd_it) > 0 else K

        # 计算 t→i（文本查询图像）的排名
        gnd_ti = (L_query[i].unsqueeze(0).mm(L_db.t()) > 0).type(torch.float).squeeze()
        if torch.sum(gnd_ti) == 0:
            continue
        hamm_ti = calc_hammingDist(qB_txt[i, :], rB_img)
        _, ind_ti = torch.sort(hamm_ti)
        ind_ti.squeeze_()
        gnd_ti = gnd_ti[ind_ti][:K]  # 只考虑前 K 个检索结果
        rank_ti = (torch.nonzero(gnd_ti)[0] + 1).item() if torch.sum(gnd_ti) > 0 else K

        # 计算 CRC@K 贡献
        crc += abs(rank_it - rank_ti) / K

    crc_k = 1 - (crc / num_query)
    return crc_k


def cos(A, B=None):
    """cosine"""
    An = normalize(A, norm='l2', axis=1)
    if (B is None) or (B is A):
        return np.dot(An, An.T)
    Bn = normalize(B, norm='l2', axis=1)
    return np.dot(An, Bn.T)


def euclidean(A, B=None, sqrt=False):
    aTb = np.dot(A, B.T)
    if (B is None) or (B is A):
        aTa = np.diag(aTb)
        bTb = aTa
    else:
        aTa = np.diag(np.dot(A, A.T))
        bTb = np.diag(np.dot(B, B.T))
    D = aTa[:, np.newaxis] - 2.0 * aTb + bTb[np.newaxis, :]
    if sqrt:
        D = np.sqrt(D)
    return D


def norm_max_min(x: torch.Tensor, dim=None):
    if dim is None:
        max = torch.max(x)
        min = torch.min(x)
    if dim is not None:
        max = torch.max(x, dim=dim)[0]
        min = torch.min(x, dim=dim)[0]
        if dim > 0:
            max = max.unsqueeze(len(x.shape) - 1)
            min = min.unsqueeze(len(x.shape) - 1)
    norm = (x - min) / (max - min)
    return norm


def norm_mean(x: torch.Tensor, dim=None):
    if dim is None:
        mean = torch.mean(x)
        std = torch.std(x)
    if dim is not None:
        mean = torch.mean(x, dim=dim)
        std = torch.std(x, dim=dim)
        if dim > 0:
            mean = mean.unsqueeze(len(x.shape) - 1)
            std = std.unsqueeze(len(x.shape) - 1)
    norm = (x - mean) / std
    return norm


def norm_abs_mean(x: torch.Tensor, dim=None):
    if dim is None:
        mean = torch.mean(x)
        std = torch.std(x)
    if dim is not None:
        mean = torch.mean(x, dim=dim)
        std = torch.std(x, dim=dim)
        if dim > 0:
            mean = mean.unsqueeze(len(x.shape) - 1)
            std = std.unsqueeze(len(x.shape) - 1)
    norm = torch.abs(x - mean) / std
    return norm


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def calc_IF(all_bow):
    word_num = torch.sum(all_bow, dim=0)
    total_num = torch.sum(word_num)
    IF = word_num / total_num
    return IF
