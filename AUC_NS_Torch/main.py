#-*- coding: utf-8 -*-

import os
import datetime
import random
import torch
import numpy as np


from parse import parse_args
from tqdm import tqdm
from data import *
from model import *
from evaluation import *
from negative_sampling import *
# print(torch.__version__)
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_path():
    directory = 'data/'
    if arg.dataset == '100k':
        total_file = directory + '/' + '100k.csv'
        train_file = directory + '/' + '100k_train.csv'
        test_file = directory + '/' + '100k_test.csv'
    elif arg.dataset == 'yahoo':
        total_file = directory + '/' + 'yahoo1.csv'
        train_file = directory + '/' + 'yahoo1_train.csv'
        test_file = directory + '/' + 'yahoo1_test.csv'
    elif arg.dataset == '1M':
        total_file = directory + '/' + '1m1.csv'
        train_file = directory + '/' + '1m1_train.csv'
        test_file = directory + '/' + '1m1_test.csv'
    return total_file, train_file, test_file


def log():
    if arg.log:
        path = arg.log_root
        # path = 'log/' + arg.dataset
        if not os.path.exists(path):
            os.makedirs(path)
        file = path + '/' + str(arg.lr_dc_epoch) + '-AUC_NS-' + str(arg.num_negsamples) + '-' + str(
                arg.M) + '--' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
        f = open(file, 'w')
        print('----------------loging----------------')
    else:
        f = sys.stdout
    return f


def get_numbers_of_ui_and_divider(file):
    '''
    :param file: data path
    :return:
    num_users: total number of users
    num_items: total number of items
    interaction_counter: total number of interactions
    item_divider: [hot item list, cold item list]
    '''
    data = pd.read_csv(file, header=0, dtype='str', sep=',')
    userlist = list(data['user'].unique())
    itemlist = list(data['item'].unique())
    popularity = np.zeros(len(itemlist))
    interaction_counter = 0
    for i in data.itertuples():
        user, item, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
        user, item = int(user), int(item)
        popularity[int(item)] += 1
        interaction_counter += 1
    num_users, num_items = len(userlist), len(itemlist)

    # Dividing HOT&COLD
    x = np.argsort(popularity)
    item_threshold = int(num_items * 0.85)
    divide_item = x[item_threshold]
    popularty_threshold = popularity[divide_item]
    Hot_item = list(np.where(popularity >= popularty_threshold)[0])
    Cold_item = list(np.where(popularity < popularty_threshold)[0])
    item_divider = [Hot_item, Cold_item]
    return num_users, num_items, interaction_counter, item_divider


def load_train_data(path, num_item):
    data = pd.read_csv(path, header=0, sep=',')
    data_dict = {}
    datapair = []
    popularity = np.zeros(num_item)
    for i in data.itertuples():
        user, item, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
        user, item = int(user), int(item)
        popularity[int(item)] += 1
        data_dict.setdefault(user, {})
        data_dict[user][item] = 1
        datapair.append((user, item))
    prior = popularity / sum(popularity)
    return data_dict, prior, datapair, popularity


def load_test_data(path, num_user, num_item):
    data = pd.read_csv(path, header=0, sep=',')
    label = np.zeros((num_user, num_item))
    data_dict = {}
    popularity = np.zeros(num_item)
    for i in data.itertuples():
        user, item, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
        user, item = int(user), int(item)
        popularity[int(item)] += 1
        data_dict.setdefault(user, set())
        data_dict[user].add(item)
        label[user, item] = 1
    return data_dict, label, popularity


def collect_G_Lap_Adj():
    G_Lap_tensor = convert_spmat_to_sptensor(dataset.Lap_mat)
    G_Adj_tensor = convert_spmat_to_sptensor(dataset.Adj_mat)
    G_Lap_tensor = G_Lap_tensor.to(device)
    G_Adj_tensor = G_Adj_tensor.to(device)
    return G_Lap_tensor, G_Adj_tensor


def get_prior_beta(prior):
    prior_beta = prior.copy()
    for i in range(len(prior_beta)):
        prior_beta[i] = pow(prior_beta[i], arg.beta)
    return prior_beta


def model_init():
    # A new train
    model_path = r'.\model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if arg.train_mode == 'new_train':
        if arg.encoder == 'MF':
            model = MF(num_users, num_items, arg, device)
        if arg.encoder == 'LightGCN':
            g_laplace, g_adj = collect_G_Lap_Adj()
            model = LightGCN(num_users, num_items, arg, device, g_laplace, g_adj)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.l2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg.lr_dc_epoch, gamma=arg.lr_dc)
        checkpoint = 0
    # Continue train
    else:
        checkpoint = torch.load(r'.\model\{}-{}--{}-{}-{}-ex_model.pth'.format(arg.dataset, arg.encoder, arg.alpha, arg.beta, arg.gama))
        if arg.encoder == 'MF':
            model = MF(num_users, num_items, arg, device)
        if arg.encoder == 'LightGCN':
            g_laplace, g_adj = collect_G_Lap_Adj()
            model = LightGCN(num_users, num_items, arg, device, g_laplace, g_adj)
        model.load_state_dict(checkpoint['net'])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.l2)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg.lr_dc_epoch, gamma=arg.lr_dc)
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('epoch_begin:', checkpoint['epoch'] + 1)
    return model, optimizer, scheduler, checkpoint


def model_train(real_epoch):
    print('-------------------------------------------', file=f)
    print('-------------------------------------------')
    print('epoch: ', real_epoch, file=f)
    print('epoch: ', real_epoch)
    print('start training: ', datetime.datetime.now(), file=f)
    print('start training: ', datetime.datetime.now())
    st = time.time()
    model.train()
    total_loss = []

    I_plus_list = dataset.pos_lens_List.to(device)
    I_minus_list = dataset.neg_lens_List.to(device)

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        # Fetch Data
        users = batch[:,0]
        items = batch[:,1]

        # To device
        batch = batch.to(device)

        # Calculate Score for users
        rating_score = model.calculate_score(users)


        # Negative Sampling
        negtives = AUC_NS(arg, users, rating_score, batch, I_plus_list, I_minus_list, prior_beta, num_items)

        # Calculate Loss
        loss = model(users, items, negtives)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    print('Loss:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']), file=f)
    print('Loss:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']))
    print('Training time:[%0.2f s]' % (time.time() - st))
    print('Training time:[%0.2f s]' % (time.time() - st), file=f)


def model_test():
    print('----------------', file=f)
    print('----------------')
    print('start predicting: ', datetime.datetime.now(), file=f)
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    sp = time.time()
    Pre_dic, Recall_dict, F1_dict, NDCG_dict, OHR_dict, UHR_dict, OCR_dict, UCR_dict,  FPR_dict, FNR_dict = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    rating_mat = model.predict()  # |U| * |V|
    if device == 'cpu':
        rating_mat = rating_mat.detach().numpy()
    else:
        rating_mat = rating_mat.cpu().detach().numpy()
    rating_mat = erase(rating_mat, train_dict)
    for k in arg.topk:
        metrices = topk_eval(rating_mat, test_label, k, item_divider, test_dict)
        precision, recall, F1, ndcg, OHR, UHR, OCR, UCR, FPR, FNR = metrices[0], metrices[1], metrices[2], metrices[3],  metrices[4], metrices[5], metrices[6], metrices[7], metrices[8], metrices[9]
        Pre_dic[k] = precision
        Recall_dict[k] = recall
        F1_dict[k] = F1
        NDCG_dict[k] = ndcg
        OHR_dict[k] = OHR
        UHR_dict[k] = UHR
        OCR_dict[k] = OCR
        UCR_dict[k] = UCR
        FPR_dict[k] = FPR
        FNR_dict[k] = FNR
    print('Predicting time:[%0.2f s]' % (time.time() - sp))
    print('Predicting time:[%0.2f s]' % (time.time() - sp), file=f)
    return Pre_dic, Recall_dict, F1_dict, NDCG_dict, OHR_dict, UHR_dict, OCR_dict, UCR_dict,  FPR_dict, FNR_dict


def print_epoch_result(real_epoch, Pre_dic, Recall_dict, F1_dict, NDCG_dict, OHR_dict, UHR_dict, OCR_dict, UCR_dict, FPR_dict, FNR_dict):
    best_result = {}
    best_epoch = {}
    for k in arg.topk:
        best_result[k] = [0., 0., 0., 0., 1., 1., 1., 1., 1., 1.]
        best_epoch[k] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for k in arg.topk:
        if Pre_dic[k] > best_result[k][0]:
            best_result[k][0], best_epoch[k][0] = Pre_dic[k], real_epoch
            best_result[k][4], best_epoch[k][4] = OHR_dict[k], real_epoch
            best_result[k][5], best_epoch[k][5] = UHR_dict[k], real_epoch
            best_result[k][6], best_epoch[k][6] = OCR_dict[k], real_epoch
            best_result[k][7], best_epoch[k][7] = UCR_dict[k], real_epoch
        if Recall_dict[k] > best_result[k][1]:
            best_result[k][1], best_epoch[k][1] = Recall_dict[k], real_epoch
        if F1_dict[k] > best_result[k][2]:
            best_result[k][2], best_epoch[k][2] = F1_dict[k], real_epoch
        if NDCG_dict[k] > best_result[k][3]:
            best_result[k][3], best_epoch[k][3] = NDCG_dict[k], real_epoch
        if FPR_dict[k] < best_result[k][8]:
            best_result[k][8], best_epoch[k][8] = FPR_dict[k], real_epoch
        if FNR_dict[k] < best_result[k][9]:
            best_result[k][9], best_epoch[k][9] = FNR_dict[k], real_epoch
        print(
            'Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f\tOHR@%02d:\t%0.4f\tUHR@%02d:\t%0.4f\tOCR@%02d:\t%0.4f\tUCR@%02d:\t%0.4f\tFPR@%02d:\t%0.4f\tFNR@%02d:\t%0.4f' %
            (k, Pre_dic[k], k, Recall_dict[k], k, F1_dict[k], k, NDCG_dict[k], k, OHR_dict[k], k, UHR_dict[k], k,
             OCR_dict[k], k, UCR_dict[k], k, FPR_dict[k], k, FNR_dict[k]))
        print(
            'Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f\tOHR@%02d:\t%0.4f\tUHR@%02d:\t%0.4f\tOCR@%02d:\t%0.4f\tUCR@%02d:\t%0.4f\tFPR@%02d:\t%0.4f\tFNR@%02d:\t%0.4f' %
            (k, Pre_dic[k], k, Recall_dict[k], k, F1_dict[k], k, NDCG_dict[k], k, OHR_dict[k], k, UHR_dict[k], k,
             OCR_dict[k], k, UCR_dict[k], k, FPR_dict[k], k, FNR_dict[k], ),
            file=f)
    return best_result, best_epoch


def print_best_result(best_result, best_epoch):
    print('------------------best result-------------------', file=f)
    print('------------------best result-------------------')
    for k in arg.topk:
        print(
            'Best Result: Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f\tOHR@%02d:\t%0.4f\tUHR@%02d:\t%0.4f\tOCR@%02d:\t%0.4f\tUCR@%02d:\t%0.4f\tFPR@%02d:\t%0.4f\tFNR@%02d:\t%0.4f\t[%0.2f s]' %
            (k, best_result[k][0], k, best_result[k][1], k, best_result[k][2], k, best_result[k][3], k,
             best_result[k][4], k, best_result[k][5], k, best_result[k][6], k, best_result[k][7], k, best_result[k][8],
             k, best_result[k][9],  (time.time() - t0)))
        print(
            'Best Result: Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f\tOHR@%02d:\t%0.4f\tUHR@%02d:\t%0.4f\tOCR@%02d:\t%0.4f\tUCR@%02d:\t%0.4f\tFPR@%02d:\t%0.4f\tFNR@%02d:\t%0.4f\t[%0.2f s]' %
            (k, best_result[k][0], k, best_result[k][1], k, best_result[k][2], k, best_result[k][3], k,
             best_result[k][4], k, best_result[k][5], k, best_result[k][6], k, best_result[k][7], k, best_result[k][8],
             k, best_result[k][9],  (time.time() - t0)), file=f)

        print(
            'Best Epoch: Pre@%02d: %d\tRecall@%02d: %d\tF1@%02d: %d\tNDCG@%02d: %d\tOHR@%02d: %d\tUHR@%02d: %d\tOCR@%02d: %d\tUCR@%02d: %d\tFPR@%02d: %d\tFNR@%02d: %d\t[%0.2f s]' % (
                k, best_epoch[k][0], k, best_epoch[k][1], k, best_epoch[k][2], k, best_epoch[k][3], k, best_epoch[k][4],
                k, best_epoch[k][5], k, best_epoch[k][6], k, best_epoch[k][7], k, best_epoch[k][8], k, best_epoch[k][9],
                (time.time() - t0)))
        print(
            'Best Epoch: Pre@%02d: %d\tRecall@%02d: %d\tF1@%02d: %d\tNDCG@%02d: %d\tOHR@%02d: %d\tUHR@%02d: %d\tOCR@%02d: %d\tUCR@%02d: %d\tFPR@%02d: %d\tFNR@%02d: %d\t[%0.2f s]' % (
                k, best_epoch[k][0], k, best_epoch[k][1], k, best_epoch[k][2], k, best_epoch[k][3], k, best_epoch[k][4],
                k, best_epoch[k][5], k, best_epoch[k][6], k, best_epoch[k][7], k, best_epoch[k][8], k, best_epoch[k][9],
                (time.time() - t0)), file=f)
    print('------------------------------------------------', file=f)
    print('------------------------------------------------')
    print('Run time: %0.2f s' % (time.time() - t0), file=f)
    print('Run time: %0.2f s' % (time.time() - t0))


if __name__ == '__main__':
    t0 = time.time()
    arg = parse_args()
    f = log()

    init_seed(2022)
    total_file, train_file, test_file = get_data_path()
    num_users, num_items, num_interaction, item_divider = get_numbers_of_ui_and_divider(total_file)

    # Load Data
    train_dict, prior, train_pair, train_popularity = load_train_data(train_file, num_items)
    test_dict, test_label, test_popularity = load_test_data(test_file, num_users, num_items)

    dataset = Data(train_pair, arg, num_users, num_items)
    train_loader = DataLoader(dataset, batch_size=arg.batch_size, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True, pin_memory=True, num_workers=arg.num_workers)

    # Init Model
    model, optimizer, scheduler, checkpoint = model_init()

    prior_beta = torch.tensor(get_prior_beta(prior)).to(device)

    # Train and Test
    for epoch in range(arg.epochs):
        if arg.train_mode == 'new_train':
            real_epoch = epoch
        else:
            real_epoch = checkpoint['epoch'] + 1 + epoch

        model_train(real_epoch)
        Pre_dic, Recall_dict, F1_dict, NDCG_dict, OHR_dict, UHR_dict, OCR_dict, UCR_dict,  FPR_dict, FNR_dict = model_test()
        scheduler.step()
        best_result, best_epoch = print_epoch_result(real_epoch, Pre_dic, Recall_dict, F1_dict, NDCG_dict, OHR_dict, UHR_dict, OCR_dict, UCR_dict,  FPR_dict, FNR_dict)
    print_best_result(best_result, best_epoch)
    f.close()

    # Save Checkpoint
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'epoch': real_epoch}
    torch.save(state, r'.\model\{}-{}--{}-{}-{}-ex_model.pth'.format(arg.dataset, arg.encoder, arg.alpha, arg.beta, arg.gama))






