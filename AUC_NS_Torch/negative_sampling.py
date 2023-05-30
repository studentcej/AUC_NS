import numpy as np
import torch


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

def gradient(x1,x2):
    return 1-torch.sigmoid(x1-x2)

def AUC_NS(arg, users, ui_scores, batch, I_plus_list, I_minus_list, prior_beta, num_items):

    pos_set_list = batch[:, 2:(2 + arg.N)] #[bs * N]
    neg_set_list = batch[:, (2 + arg.N):(2 + 2 * arg.N)] #[bs * N]
    candidate_set_list = batch[:, (2 + 2 * arg.N):] #[bs * M]
    batch_ranting_vectors = ui_scores  #[bs * |I|]



    batch_extrapos_scores = batch_ranting_vectors[torch.arange(batch.shape[0]).unsqueeze(-1).expand(batch.shape[0],arg.N),pos_set_list] #[bs * N]
    batch_extraneg_scores = batch_ranting_vectors[torch.arange(batch.shape[0]).unsqueeze(-1).expand(batch.shape[0],arg.N),neg_set_list] #[bs * N]
    candidate_scores = batch_ranting_vectors[torch.arange(batch.shape[0]).unsqueeze(-1).expand(batch.shape[0],arg.M),candidate_set_list] #[bs * M]


    info_plus_mean = gradient(batch_extrapos_scores.unsqueeze(1), candidate_scores.unsqueeze(-1)).mean(dim=-1) # [batch * M]
    info_minus_mean = gradient(candidate_scores.unsqueeze(-1), batch_extraneg_scores.unsqueeze(1)).mean(dim=-1) # [batch * M]

    info_plus_list = info_plus_mean * I_plus_list[users].unsqueeze(-1)
    info_minus_list = arg.gama * info_minus_mean * I_minus_list[users].unsqueeze(-1)

    # Step 2 : computing prior probability
    p_fn = prior_beta[candidate_set_list]  # O(1) tau_plus

    # Step 3 : computing empirical distribution function (likelihood)
    F_n = (batch_ranting_vectors.unsqueeze(1) <= candidate_scores.unsqueeze(-1)).sum(dim=-1)/(num_items + 1)

    # Step 4: computing posterior probability
    negdist = (2 * (1 - F_n) * arg.alpha + 2 * F_n * (1 - arg.alpha)) * p_fn
    posdist = (2 * (1 - F_n) * (1 - arg.alpha) + 2 * F_n * arg.alpha) * (1 - p_fn)
    unbias = negdist / (negdist + posdist)

    # Step 5: computing conditional sampling risk
    AUC_gain = unbias * info_plus_list - (1 - unbias) * info_minus_list  # O(1)

    # Sampling
    j = candidate_set_list[torch.arange(batch.shape[0]).unsqueeze(-1),AUC_gain.topk(k=arg.num_negsamples, dim=-1).indices]
    return j