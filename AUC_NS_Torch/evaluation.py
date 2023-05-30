import math
import heapq
import numpy as np


# evaluation
def erase(score, train_dict):
    for user in train_dict:
        for item in train_dict[user]:
            score[user, item] = -1000.0
    return score


def topk_eval(score, label, k, item_devider, test_dict):
    '''
    :param score: prediction
    :param k: number of top-k
    '''
    evaluation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    counter = 0

    rec_list = []
    discountlist = [1 / math.log(i + 1, 2) for i in range(1, k + 1)]

    for user_no in range(score.shape[0]):

        user_score = score[user_no].tolist()
        user_label = label[user_no].tolist()
        label_count = int(sum(user_label))
        topn_recommend_score = heapq.nlargest(k, user_score)
        topn_recommend_index = [user_score.index(i) for i in
                                topn_recommend_score]  # map(user_score.index,topn_recommend_score)
        rec_list.append(topn_recommend_index)
        topn_recommend_label = [user_label[i] for i in topn_recommend_index]
        idcg = discountlist[0:label_count]

        # FPR FNR
        test_item = set(test_dict[user_no])
        rec_item = set(topn_recommend_index)
        underestimate_items = test_item - rec_item
        overestimate_items = rec_item - test_item
        FPR = len(overestimate_items)/len(rec_item)
        FNR = len(underestimate_items)/len(test_item)
        underestimate_list = list(underestimate_items)
        error_list = list(overestimate_items)

        # Popularity Bias
        rec_cold = 0
        rec_hot = 0
        test_cold = 0
        test_hot = 0
        overestimate_cold = 0
        overestimate_hot = 0
        underestimate_cold = 0
        underestimate_hot = 0

        for i in rec_list[user_no]:
            if i in item_devider[1]:
                rec_cold += 1
            elif i in item_devider[0]:
                rec_hot += 1

        for i in test_dict[user_no]:
            if i in item_devider[1]:
                test_cold += 1
            elif i in item_devider[0]:
                test_hot += 1

        for i in underestimate_list:
            if i in item_devider[1]:
                underestimate_cold += 1
            elif i in item_devider[0]:
                underestimate_hot += 1

        for i in error_list:
            if i in item_devider[1]:
                overestimate_cold += 1
            elif i in item_devider[0]:
                overestimate_hot += 1

        if label_count == 0:
            counter += 1
            continue
        else:
            topk_label = topn_recommend_label[0:k]
            true_positive = sum(topk_label)
            evaluation[0] += true_positive / k                                  # Precision
            evaluation[1] += true_positive / label_count                        # Recall
            evaluation[2] += 2 * true_positive / (k + label_count)              # F1
            evaluation[3] += np.dot(topk_label, discountlist[0:]) / sum(idcg)   # NDCG
            if rec_hot:
                evaluation[4] += overestimate_hot / rec_hot                     # OHR
            if test_hot:
                evaluation[5] += underestimate_hot / test_hot                     # UHR
            if rec_cold:
                evaluation[6] += overestimate_cold / rec_cold                   # OCR
            if test_cold:
                evaluation[7] += underestimate_cold / test_cold                   # UCR
            evaluation[8] += FPR
            evaluation[9] += FNR
    return [i / (score.shape[0] - counter) for i in evaluation]




