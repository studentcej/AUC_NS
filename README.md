## Brief Introduction
This thesis presents AUC-optimal negative sampling for implicit collaborative filtering, which aims to address the issue of popularity bias by achieving a high true positive rate within a specific range of false positive rates.
## Prerequisites
- Python 3.8 
- PyTorch 1.11.0

## Some Tips
Flags in `parse.py`:

Model training related settings:

- `--train_mode` Choosing to either start a new training session, or continue training with the model saved from your previous session.
- `--encoder` Choosing MF or LightGCTN as the Backbone of the CF model.
- `--epochs` Number of sweeps over the dataset to train.
- `--dataset` Choosing 100k, yahoo, 1M, or your dataset.

You can set the relevant parameters for model training,

- `--batch_size` size of each batch
- `--l2` l2 regulation constant.
- `--lr` learning rate.
- `--lr_dc` learning rate decay rate.
- `--lr_dc_epoch` training epoch at which the learning rate starts to decay.

Suggested training parameters parameters are:
#### Suggesting Model Training Parameters
|                | batch_size |  l2   |  lr  | lr_dc | lr_dc_epoch  | 
|----------------|:----------:|:-----:|:----:|:-----:|:------------:|
| 100k-MF        |    128     | 1e-4  | 0.1  |  0.1  |  [20,60,80]  |
| 100k-LightGCN  |    128     | 1e-5  | 0.1  |  0.1  |  [20,60,80]  |
| 1M-MF          |    128     | 1e-5  | 5e-4 |   1   |      []      |
| 1M-LightGCN    |    1024    |   0   | 1e-3 |   1   |      []      |
| yahoo-MF       |    128     |   0   | 5e-4 |   1   |      []      |
| yahoo-LightGCN |    128     |   0   | 5e-4 |   1   |      []      |

AUC-optimal negative sampling related parameters:

- `--alpha` denote the probability that the decision function assigns a higher score to a positive example than a negative example,which corresponds to macro-AUC of encoder that accounts for model-dependent bias correction.


- `--beta` modifies the prior information to reduce the model-independent bias. Specifically, it is the concentration parameter that controls the probability of popular items being true negatives.

- `--gama` controls the range of false positive rates that we aim to achieve a high true positive rate within. For top-k evaluation tasks, a smaller value of gama should be chosen as the value of k decreases.

- `--N` size of the additional positive and negative samples used to estimate the sum of gradients. Larger N can provide more  accurate AUC information at the cost of longer training times.We fixed this parameter to 10 in our experiments.

Suggested AUC_NS parameters are:
#### Suggested AUC_NS Parameters
|                | $\alpha$ | $\beta$  | $\gamma$ |
|----------------|:--------:|:--------:|:--------:|
| 100k-MF        |   0.75   |   0.01   |  0.006   | 
| 100k-LightGCN  |   0.75   |   0.01   |  0.006   |
| 1M-MF          |   0.75   |   0.01   |  0.006   |
| 1M-LightGCN    |   0.75   |   0.01   |  0.006   |
| yahoo-MF       |   0.65   |  0.001   |  0.004   |
| yahoo-LightGCN |   0.65   |  0.001   |  0.004   |

For instance, execute the following command to train CF model using AUC_NS method.
```
python main.py --alpha 0.75 --beta 0.01 --gama 0.006
```
## AUC-NS with different loss functions.
The AUC-NS method can extend negative sampling from one to N negative examples, making it applicable to the InfoNCE loss, where it has shown excellent results on multisampling experiments, you can execute
```
python main.py --Loss Info_NCE --num_negsamples 3
```
to obtain better performance models.
