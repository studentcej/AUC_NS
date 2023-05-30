import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder MF
class MF(nn.Module):
    def __init__(self, num_users, num_items, arg, device):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.arg = arg
        self.device = device

        self.dim = arg.dim

        self.User_Emb = nn.Embedding(self.num_users, self.dim)
        nn.init.xavier_normal_(self.User_Emb.weight)
        self.Item_Emb = nn.Embedding(self.num_items, self.dim)
        nn.init.xavier_normal_(self.Item_Emb.weight)

    def computer(self):
        users_emb = self.User_Emb.weight
        items_emb = self.Item_Emb.weight
        return users_emb, items_emb

    def forward(self, users, items, negatives):

        # Fetch Emb
        all_users_emb , all_items_emb = self.computer()
        users_emb = all_users_emb[users]
        items_emb = all_items_emb[items]
        neg_item_embs = all_items_emb[negatives]  # bs * k * d

        # Calculate Pos
        pos_scores = torch.mul(users_emb, items_emb)
        pos_scores = pos_scores.sum(dim=1)  # (bs,)
        pos_exp = torch.exp(pos_scores)

        # Calculate Neg
        neg_k = (users_emb * neg_item_embs.view(self.arg.num_negsamples, self.arg.batch_size, self.dim)).sum(dim=-1)


        # Calculate Loss
        if self.arg.LOSS == 'Info_NCE':
            neg_exp = torch.exp(neg_k).sum(dim=0)
            # InfoNCE_loss
            InfoNCE_loss = (- torch.log(pos_exp / (pos_exp + neg_exp))).mean()
            return InfoNCE_loss
        elif self.arg.LOSS == 'BPR' and self.arg.num_negsamples == 1:
            neg_scores = neg_k.squeeze()
            BPR_loss = torch.mean(F.softplus(neg_scores - pos_scores))
            return BPR_loss
        else:
            print('Parameters Wrong')
            sys.exit()

    def predict(self):
        all_users_emb = self.User_Emb.weight
        all_items_emb = self.Item_Emb.weight
        rate_mat = torch.mm(all_users_emb, all_items_emb.t())
        return rate_mat

    def calculate_score(self, users):
        all_users_emb = self.User_Emb.weight
        all_items_emb = self.Item_Emb.weight
        users_emb = all_users_emb[users]
        rate_score = torch.mm(users_emb, all_items_emb.t())
        return rate_score


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, arg, device, g_laplace, g_adj):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.g_laplace = g_laplace
        self.g_adj = g_adj
        self.arg = arg
        self.device = device
        self.dim = arg.dim
        self.hop = arg.hop

        self.User_Emb = nn.Embedding(self.num_users, self.dim)
        nn.init.xavier_normal_(self.User_Emb.weight)
        self.Item_Emb = nn.Embedding(self.num_items, self.dim)
        nn.init.xavier_normal_(self.Item_Emb.weight)

        # LightGCN Agg
        self.global_agg = []
        for i in range(self.hop):
            agg = LightGCNAgg(self.dim)
            self.add_module('Agg_LightGCN_{}'.format(i), agg)
            self.global_agg.append(agg)

    def computer(self):
        users_emb = self.User_Emb.weight
        items_emb = self.Item_Emb.weight
        all_emb = torch.cat((users_emb, items_emb), dim=0)
        embs = [all_emb]
        for i in range(self.hop):
            aggregator = self.global_agg[i]
            x = aggregator(A=self.g_laplace, x=embs[i])
            embs.append(x)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items



    def forward(self, users, items, negatives):
        # Fetch Emb
        all_users_emb, all_items_emb = self.computer()
        users_emb = all_users_emb[users]
        items_emb = all_items_emb[items]
        neg_item_embs = all_items_emb[negatives]  # bs * k * d

        # Calculate Pos
        pos_scores = torch.mul(users_emb, items_emb)
        pos_scores = pos_scores.sum(dim=1)  # (bs,)
        pos_exp = torch.exp(pos_scores)

        # Calculate Neg
        neg_k = (users_emb * neg_item_embs.view(self.arg.num_negsamples, self.arg.batch_size, self.dim)).sum(dim=-1)

        # Calculate Loss
        if self.arg.LOSS == 'Info_NCE':
            neg_exp = torch.exp(neg_k).sum(dim=0)
            # InfoNCE_loss
            InfoNCE_loss = (- torch.log(pos_exp / (pos_exp + neg_exp))).mean()
            return InfoNCE_loss
        elif self.arg.LOSS == 'BPR' and self.arg.num_negsamples == 1:
            neg_scores = neg_k.squeeze()
            BPR_loss = torch.mean(F.softplus(neg_scores - pos_scores))
            return BPR_loss
        else:
            print('Parameters Wrong')
            sys.exit()



    def predict(self):
        all_users_emb, all_items_emb = self.computer()      # |U| * d, |V| * d
        rate_mat = torch.mm(all_users_emb, all_items_emb.t())
        return rate_mat

    def calculate_score(self, users):
        all_users_emb = self.User_Emb.weight
        all_items_emb = self.Item_Emb.weight
        users_emb = all_users_emb[users]
        rate_score = torch.mm(users_emb, all_items_emb.t())
        return rate_score


class LightGCNAgg(nn.Module):
    def __init__(self, hidden_size):
        super(LightGCNAgg, self).__init__()
        self.dim = hidden_size

    def forward(self, A, x):
        '''
            A: n \times n
            x: n \times d
        '''
        return torch.sparse.mm(A, x)
