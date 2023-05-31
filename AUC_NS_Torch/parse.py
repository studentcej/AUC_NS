import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Log Information
    parser.add_argument('--log_root', default=r'.\result')
    parser.add_argument('--log', type=bool, default=True)

    # Random Seed
    parser.add_argument('--seed', type=int, default=2022)

    # Training Args
    parser.add_argument('--train_mode', default='new_train', help='training mode:new_train continue_train')
    parser.add_argument('--encoder', default='MF', help='MF LightGCN')
    parser.add_argument('--epochs', type=int, default=150)  # 150 200 300
    parser.add_argument('--batch_size', type=int, default=128)  # 128 1024
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 penalty')  # 1e-4 1e-5 0
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')  # 0.1 5e-4 1e-3
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate') # 0.1 1
    parser.add_argument('--lr_dc_epoch', type=list, default=[20, 60, 80], help='the epoch which the learning rate decay')  # 20 60 80
    parser.add_argument('--LOSS', default='BPR', help='loss')  # BPR, Info_NCE
    parser.add_argument('--num_workers', type=int, default=4)  # Speed up training
    parser.add_argument('--dim', type=int, default=32, help='dimension of vector')  # Dim of encoders
    parser.add_argument('--hop', type=int, default=3, help='number of LightGCN layers')  # Hop of GCN

    # Dataset
    parser.add_argument('--dataset', default='100k', help='dataset')  # 100k yahoo movielens

    # Sampling Args
    parser.add_argument('--M', type=int, default=5, help='size of candidate set')
    parser.add_argument('--num_negsamples', type=int, default=1, help='number of negative samples for each NS')

    # AUC_NS Args
    parser.add_argument('--N', type=int, default=10, help='amount of extra plus and extra minus')
    parser.add_argument('--alpha', type=float, default=0.75, help='AUC of encoders')
    parser.add_argument('--beta', type=int, default=0.01, help='popularity_punish_rate')
    parser.add_argument('--gama', type=float, default=0.006, help='partialAUC')

    # Evaluation Arg
    parser.add_argument('--topk', type=list, default=[5, 10, 20], help='length of recommendation list')

    return parser.parse_args()


