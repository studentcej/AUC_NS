import math
import heapq
import numpy as np
import torch


# evaluation

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


def get_rec_tensor(k, topn_rec_index, num_items):
    index0 = torch.arange(topn_rec_index.shape[0]).to(device)
    index0 = index0.unsqueeze(-1).expand(topn_rec_index.shape[0], k)
    rec_tensor = torch.zeros(topn_rec_index.shape[0],num_items).to(device)
    rec_tensor[index0, topn_rec_index] = 1
    return rec_tensor, index0


def get_idcg(discountlist, label_count_all, k):
    idcg = torch.zeros(len(label_count_all)).to(device)
    label_count_list = label_count_all.tolist()
    for i in range(len(label_count_all)):
        idcg[i] = discountlist[0:int(label_count_list[i])].sum()
    return idcg


def topk_eval(score,  k, test_tensor, hot_tensor):
    '''
    :param score: prediction
    :param k: number of top-k
    '''
    evaluation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # counter = 0

    # rec_list = []


    label_count_all = test_tensor.sum(dim=-1)
    label_none_zero = torch.count_nonzero(label_count_all)

    topn_rec_index = score.topk(k=k, dim=-1).indices
    rec_tensor, index0 = get_rec_tensor(k, topn_rec_index, score.shape[1])
    hit_tensor = rec_tensor * test_tensor
    true_positive_all = hit_tensor.sum(dim=-1)

    # discountlist = [1 / math.log(i + 1, 2) for i in range(1, k + 1)]
    discountlist = torch.tensor([1 / math.log(i + 1, 2) for i in range(1, k + 1)]).to(device)
    dcg = (hit_tensor[index0, topn_rec_index] * discountlist).sum(dim=-1)
    idcg = get_idcg(discountlist, label_count_all, k)

    pre = true_positive_all.sum(dim=-1) / k
    re = true_positive_all / label_count_all
    recall = torch.where(torch.isnan(re), torch.full_like(re, 0), re).sum(dim=-1)
    f1 = (2 * true_positive_all / label_count_all.add(k)).sum(dim=-1)
    ndcg = (dcg / idcg).sum(dim=-1)

    estimate_tensor = rec_tensor - test_tensor
    overestimate_tensor = torch.where(estimate_tensor == 1, 1, 0)
    underestimate_tensor = torch.where(estimate_tensor == -1, 1, 0)

    fpr = (overestimate_tensor.sum(dim=-1) / k).sum(dim=-1)
    fn = underestimate_tensor.sum(dim=-1) / label_count_all
    fnr = torch.where(torch.isnan(fn), torch.full_like(fn, 0), fn).sum(dim=-1)

    over_hot_tensor = overestimate_tensor * hot_tensor
    over_cold_tensor = overestimate_tensor * (1-hot_tensor)
    under_hot_tensor = underestimate_tensor * hot_tensor
    under_cold_tensor = underestimate_tensor * (1-hot_tensor)

    rec_hot = rec_tensor * hot_tensor
    rec_cold = rec_tensor * (1-hot_tensor)
    test_hot = test_tensor * hot_tensor
    test_cold = test_tensor * (1-hot_tensor)

    oh = (over_hot_tensor.sum(dim=-1) / rec_hot.sum(dim=-1))
    uh = (under_hot_tensor.sum(dim=-1) / test_hot.sum(dim=-1))
    oc = (over_cold_tensor.sum(dim=-1) / rec_cold.sum(dim=-1))
    uc = (under_cold_tensor.sum(dim=-1) / test_cold.sum(dim=-1))

    ohr = torch.where(torch.isnan(oh), torch.full_like(oh, 0), oh).sum(dim=-1)
    uhr = torch.where(torch.isnan(uh), torch.full_like(uh, 0), uh).sum(dim=-1)
    ocr = torch.where(torch.isnan(oc), torch.full_like(oc, 0), oc).sum(dim=-1)
    ucr = torch.where(torch.isnan(uc), torch.full_like(uc, 0), uc).sum(dim=-1)


    # for user_no in range(score.shape[0]):
    #
    #     user_score = score[user_no].tolist()
    #     user_label = label[user_no].tolist()
    #     label_count = int(sum(user_label))
    #     topn_recommend_score = heapq.nlargest(k, user_score)
    #     topn_recommend_index = [user_score.index(i) for i in
    #                             topn_recommend_score]  # map(user_score.index,topn_recommend_score)
    #     rec_list.append(topn_recommend_index)
    #     topn_recommend_label = [user_label[i] for i in topn_recommend_index]
    #     idcg = discountlist[0:label_count]
    #
    #     # FPR FNR
    #     test_item = set(test_dict[user_no])
    #     rec_item = set(topn_recommend_index)
    #     underestimate_items = test_item - rec_item
    #     overestimate_items = rec_item - test_item
    #     FPR = len(overestimate_items)/len(rec_item)
    #     FNR = len(underestimate_items)/len(test_item)
    #     underestimate_list = list(underestimate_items)
    #     error_list = list(overestimate_items)
    #
    #     # Popularity Bias
    #     rec_cold = 0
    #     rec_hot = 0
    #     test_cold = 0
    #     test_hot = 0
    #     overestimate_cold = 0
    #     overestimate_hot = 0
    #     underestimate_cold = 0
    #     underestimate_hot = 0
    #
    #     for i in rec_list[user_no]:
    #         if i in item_devider[1]:
    #             rec_cold += 1
    #         elif i in item_devider[0]:
    #             rec_hot += 1
    #
    #     for i in test_dict[user_no]:
    #         if i in item_devider[1]:
    #             test_cold += 1
    #         elif i in item_devider[0]:
    #             test_hot += 1
    #
    #     for i in underestimate_list:
    #         if i in item_devider[1]:
    #             underestimate_cold += 1
    #         elif i in item_devider[0]:
    #             underestimate_hot += 1
    #
    #     for i in error_list:
    #         if i in item_devider[1]:
    #             overestimate_cold += 1
    #         elif i in item_devider[0]:
    #             overestimate_hot += 1
    #
    #     if label_count == 0:
    #         counter += 1
    #         continue
    #     else:
    #         topk_label = topn_recommend_label[0:k]
    #         true_positive = sum(topk_label)
    #         evaluation[0] += true_positive / k                                  # Precision
    #         evaluation[1] += true_positive / label_count                        # Recall
    #         evaluation[2] += 2 * true_positive / (k + label_count)              # F1
    #         evaluation[3] += np.dot(topk_label, discountlist[0:]) / sum(idcg)   # NDCG
    #         if rec_hot:
    #             evaluation[4] += overestimate_hot / rec_hot                     # OHR
    #         if test_hot:
    #             evaluation[5] += underestimate_hot / test_hot                     # UHR
    #         if rec_cold:
    #             evaluation[6] += overestimate_cold / rec_cold                   # OCR
    #         if test_cold:
    #             evaluation[7] += underestimate_cold / test_cold                   # UCR
    #         evaluation[8] += FPR
    #         evaluation[9] += FNR
    evaluation[0], evaluation[1], evaluation[2], evaluation[3], evaluation[4], evaluation[5], evaluation[6], evaluation[7], evaluation[8], evaluation[9] = pre.item(), recall.item(), f1.item(), ndcg.item(), ohr.item(), uhr.item(), ocr.item(), ucr.item(), fpr.item(), fnr.item()
    return [i / label_none_zero.item() for i in evaluation]



