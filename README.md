# Sub-FedAvg
Personalized Federated Learning by Structured and Unstructured Pruning under Data Heterogeneity

This repository contains the pytorch official implementation for the following paper<br>
[**Personalized Federated Learning by Structured and Unstructured Pruning under Data Heterogeneity**](https://arxiv.org/abs/2105.00562)<br>
Saeed Vahidian*, [Mahdi Morafah*](https://www.linkedin.com/in/mahdi-morafah-ab97a8106/), and Bill Lin <br>
41th IEEE International Conference on Distributed Computing Systems (Won ICDCS Conference Award) (*equal contribution) <br>
[**YouTube Presentation**](https://www.youtube.com/watch?v=ttY7T8W5YQE) <br>

If you find our repository and paper useful, please cite our work: 
```
@article{vahidian2021personalized,
  title={Personalized Federated Learning by Structured and Unstructured Pruning under Data Heterogeneity},
  author={Vahidian, Saeed and Morafah, Mahdi and Lin, Bill},
  journal={arXiv preprint arXiv:2105.00562},
  year={2021}
}
```

## Usage 
1. Glone the repository 
```bash
git clone https://github.com/MMorafah/Sub-FedAvg.git
```
1. For Running 
  * Sub-FedAvg (Hybrid) 
  ```
  sh script_s.sh 
  ```
  * Sub-FedAvg (Unstructured) 
  ```
  sh script_u.sh 
  ```
## Dependencies
torch v0.3.1, torchvision v0.2.0

### Options 
1.General federated options
```
parser.add_argument('--rounds', type=int, default=300, help="rounds of training")
parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
parser.add_argument('--bs', type=int, default=128, help="test batch size")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
parser.add_argument('--warmup_epoch', type=int, default=0, help="the number of pretrain local ep")
```
1. Model options
```
# model arguments
parser.add_argument('--model', type=str, default='lenet5', help='model name')
parser.add_argument('--ks', type=int, default=5, help='kernel size to use for convolutions')
parser.add_argument('--in_ch', type=int, default=3, help='input channels of the first conv layer')
```
1. dataset partitioning options
```
# dataset partitioning arguments
parser.add_argument('--dataset', type=str, default='cifar10', 
                    help="name of dataset: mnist, cifar10, cifar100")
parser.add_argument('--nclass', type=int, default=2, help="classes or shards per user")
parser.add_argument('--nsample_pc', type=int, default=250, 
                    help="number of samples per class or shard for each client")
parser.add_argument('--noniid', action='store_true', help='whether i.i.d or not')
parser.add_argument('--shard', action='store_true', help='whether non-i.i.d based on shard or not')
parser.add_argument('--label', action='store_true', help='whether non-i.i.d based on label or not')
parser.add_argument('--split_test', action='store_true', 
                    help='whether split test set in partitioning or not')
```
1. Structured (Hybrid) pruning options (main_s.py)
```
# pruning arguments 
parser.add_argument('--pruning_percent_ch', type=float, default=0.45, 
                    help="Pruning percent for channels (0-1)")
parser.add_argument('--pruning_percent_fc', type=float, default=10, 
                    help="Pruning percent for fully connected layers (0-100)")
parser.add_argument('--pruning_target', type=int, default=90, 
                    help="Total Pruning target percentage (0-100)")
parser.add_argument('--dist_thresh_ch', type=float, default=0.01, 
                    help="threshold for channels masks difference ")
parser.add_argument('--dist_thresh_fc', type=float, default=0.0005, 
                    help="threshold for fcs masks difference ")
parser.add_argument('--acc_thresh', type=int, default=50, 
                    help="accuracy threshold to apply the derived pruning mask")

parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                     help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001, 
                    help='scale sparse rate (default: 0.0001)')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                metavar='W', help='weight decay (default: 1e-4)')
```
1. Unstructured pruning options (main_u.py)
```
parser.add_argument('--pruning_percent', type=float, default=10, 
                        help="Pruning percent for layers (0-100)")
parser.add_argument('--pruning_target', type=int, default=30, 
                  help="Total Pruning target percentage (0-100)")
parser.add_argument('--dist_thresh_fc', type=float, default=0.0001, 
                  help="threshold for fcs masks difference ")
parser.add_argument('--acc_thresh', type=int, default=50, 
                  help="accuracy threshold to apply the derived pruning mask")
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
              metavar='W', help='weight decay (default: 1e-4)')
```
1. Other options
```
# other arguments 
parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('--is_print', action='store_true', help='verbose print')
parser.add_argument('--print_freq', type=int, default=100, help="printing frequency during training rounds")
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--load_initial', type=str, default='', help='define initial model path')

```

## Contact
- Mahdi Morafah (mmorafah@ucsd.edu)
- Saeed Vahidian (svahidia@eng.ucsd.edu)
