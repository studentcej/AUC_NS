import math
import torch
# evaluation

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


def get_rec_tensor(k, topn_rec_index, num_items):
    '''
    :param k: Top-k
    :param topn_rec_index: [|U|*k]recommended item id
    :param num_items: The total number of numbers
    :return:
    rec_tensor: [|U|*|I|] with 0/1 elements ,1 indicates the item is recommended to the user
    index_dim0:[|U|*k] dim0 index for slicing
    '''
    index_dim0 = torch.arange(topn_rec_index.shape[0]).to(device)
    index_dim0 = index_dim0.unsqueeze(-1).expand(topn_rec_index.shape[0], k)
    rec_tensor = torch.zeros(topn_rec_index.shape[0],num_items).to(device)
    rec_tensor[index_dim0, topn_rec_index] = 1
    return rec_tensor, index_dim0


def get_idcg(discountlist, test_count, k):
    idcg = torch.zeros(len(test_count)).to(device)
    label_count_list = test_count.tolist()
    for i in range(len(test_count)):
        idcg[i] = discountlist[0:int(label_count_list[i])].sum()
    return idcg


def topk_eval(score,  k, test_tensor, hot_tensor):
    '''
    :param score: prediction
    :param k: number of top-k
    '''
    evaluation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    test_count = test_tensor.sum(dim=-1)                # The number of interactions of each user in test set
    test_user_count = torch.count_nonzero(test_count)   # The number of users with interactions in the test set

    topn_rec_index = score.topk(k=k, dim=-1).indices
    rec_tensor, index_dim0 = get_rec_tensor(k, topn_rec_index, score.shape[1])
    hit_tensor = rec_tensor * test_tensor               # True positive [|U|*|I|]
    true_positive = hit_tensor.sum(dim=-1)              # [|U|,] The number of TP of each user

    discountlist = torch.tensor([1 / math.log(i + 1, 2) for i in range(1, k + 1)]).to(device) # Discount list to calculate dcg
    rec_label = hit_tensor[index_dim0, topn_rec_index]                                        # [|U|*k] The label of recommended item
    dcg = (rec_label * discountlist).sum(dim=-1)
    idcg = get_idcg(discountlist, test_count, k)

    pre = true_positive.sum(dim=-1) / k
    recall = (true_positive / (test_count + 1e-8)).sum(dim=-1)
    f1 = (2 * true_positive / test_count.add(k)).sum(dim=-1)
    ndcg = (dcg / idcg).sum(dim=-1)

    difference = rec_tensor - test_tensor
    overestimate_tensor = torch.where(difference == 1, 1, 0) # The difference with element=1 indicating the item is overestimated
    underestimate_tensor = torch.where(difference == -1, 1, 0) # The difference with element=-1 indicating the item is underestimated

    fpr = (overestimate_tensor.sum(dim=-1) / k).sum(dim=-1)
    fnr = (underestimate_tensor.sum(dim=-1) / (test_count + 1e-8)).sum(dim=-1)

    rec_hot = rec_tensor * hot_tensor           # [|U|*|I|] Recommended hot item
    rec_cold = rec_tensor * (1 - hot_tensor)
    test_hot = test_tensor * hot_tensor
    test_cold = test_tensor * (1 - hot_tensor)

    over_hot_tensor = overestimate_tensor * hot_tensor      # [|U|*|I|] Overestimated hot item
    over_cold_tensor = overestimate_tensor * (1-hot_tensor)
    under_hot_tensor = underestimate_tensor * hot_tensor
    under_cold_tensor = underestimate_tensor * (1-hot_tensor)

    ohr = (over_hot_tensor.sum(dim=-1) / (rec_hot.sum(dim=-1) + 1e-8)).sum(dim=-1)
    uhr = (under_hot_tensor.sum(dim=-1) / (test_hot.sum(dim=-1) + 1e-8)).sum(dim=-1)
    ocr = (over_cold_tensor.sum(dim=-1) / (rec_cold.sum(dim=-1) + 1e-8)).sum(dim=-1)
    ucr = (under_cold_tensor.sum(dim=-1) / (test_cold.sum(dim=-1) + 1e-8)).sum(dim=-1)


    evaluation[0], evaluation[1], evaluation[2], evaluation[3], evaluation[4], evaluation[5], evaluation[6], evaluation[7], evaluation[8], evaluation[9] = pre.item(), recall.item(), f1.item(), ndcg.item(), ohr.item(), uhr.item(), ocr.item(), ucr.item(), fpr.item(), fnr.item()
    return [i / test_user_count.item() for i in evaluation]




