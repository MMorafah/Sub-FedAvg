import numpy as np

import copy
import os 
import gc 

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data import *
from src.models import *
from src.pruning import *
from src.sub_fedavg import * 
from src.client import * 
from src.utils.options_u import args_parser 

args = args_parser()

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

torch.cuda.set_device(args.gpu) ## Setting cuda on GPU 
## Data partitioning section 

if args.dataset == 'cifar10':
    data_dir = '../data/cifar10/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
    
    nclass_cifar10 = args.nclass
    nsamples_cifar10 = args.nsample_pc
    
    if args.noniid: 
        if args.shard:
            print(f'--CIFAR-10 Non-IID-- {args.nclass} random Shards, Sample per shard {args.nsample_pc}')
            user_groups_train, user_groups_test = noniid_shard(args.dataset, train_dataset, test_dataset, 
                            args.num_users, nclass_cifar10, nsamples_cifar10, args.split_test)
        
        elif args.label: 
            print(f'--CIFAR-10 Non-IID-- {args.nclass} random Label, Sample per label {args.nsample_pc}')
            user_groups_train, user_groups_test = \
            noniid_label(args.dataset, train_dataset, test_dataset, args.num_users, nclass_cifar10,
                                 nsamples_cifar10, args.split_test)
            
        else: 
            exit('Error: unrecognized partitioning type')
    else: 
        print(f'--CIFAR-10 IID-- Split Test {args.split_test}')
        user_groups_train, user_groups_test = \
        iid(args.dataset, train_dataset, test_dataset, args.num_users, args.split_test)
            
elif args.dataset == 'cifar100':
    data_dir = '../data/cifar100/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])
    
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=apply_transform)
    
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=apply_transform)
    
    nclass_cifar100 = args.nclass
    nsamples_cifar100 = args.nsample_pc
    
    if args.noniid: 
        if args.shard:
            print(f'--CIFAR-100 Non-IID-- {args.nclass} random Shards, Sample per shard {args.nsample_pc}')
            user_groups_train, user_groups_test = noniid_shard(args.dataset, train_dataset, test_dataset, 
                        args.num_users, nclass_cifar100, nsamples_cifar100, args.split_test)
            
        elif args.label: 
            print(f'--CIFAR-100 Non-IID-- {args.nclass} random Labels, Sample per label {args.nsample_pc}')
            user_groups_train, user_groups_test = \
            noniid_label(args.dataset, train_dataset, test_dataset, args.num_users, nclass_cifar100,
                                 nsamples_cifar100, args.split_test)
        else: 
            exit('Error: unrecognized partitioning type')
    else:
        print(f'--CIFAR-100 IID-- Split Test {args.split_test}')
        user_groups_train, user_groups_test = \
        iid(args.dataset, train_dataset, test_dataset, args.num_users, args.split_test)
            
elif args.dataset == 'mnist': 
    data_dir = '../data/mnist/'
    apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
    
    nclass_mnist = args.nclass
    nsamples_mnist = args.nsample_pc
    
    if args.noniid: 
        if args.shard:
            print(f'--MNIST Non-IID-- {args.nclass} random Shards, Sample per shard {args.nsample_pc}')
            user_groups_train, user_groups_test = noniid_shard(args.dataset, train_dataset, test_dataset, 
                            args.num_users, nclass_mnist, nsamples_mnist, args.split_test)
        elif args.label: 
            print(f'--MNIST Non-IID-- {args.nclass} random Labels, Sample per label {args.nsample_pc}')
            user_groups_train, user_groups_test = \
            noniid_label(args.dataset, train_dataset, test_dataset, args.num_users, nclass_mnist,
                                 nsamples_mnist, args.split_test)        
        else: 
            exit('Error: unrecognized partitioning type')
    else: 
        print(f'--MNIST IID-- Split Test {args.split_test}')
        user_groups_train, user_groups_test = \
        iid(args.dataset, train_dataset, test_dataset, args.num_users, args.split_test)
        
## 
## Checking the partitions (total sample and labels for each client)

users_train_labels = {i: [] for i in range(args.num_users)}
users_test_labels = {i: [] for i in range(args.num_users)}

train_targets = np.array(train_dataset.targets)
test_targets = np.array(test_dataset.targets)

for i in range(args.num_users):
    ## Train Data for Each Client 
    train_count_per_client = 0 
    label = train_targets[user_groups_train[i]]
    train_count_per_client += len(label)
    label = set(label)
    users_train_labels[i] = list(label)
    
    # Test Data for Each Client 
    test_count_per_client = 0 
    label = test_targets[user_groups_test[i]]
    test_count_per_client += len(label)
    label = set(label)
    users_test_labels[i] = list(label) 
    
    #print(f'Client: {i}, Train Labels: {users_train_labels[i]}, Test Labels: {users_test_labels[i]},'
          #f' Num Train: {train_count_per_client}, Num Test: {test_count_per_client}')
        
## 
# build model
print(f'MODEL: {args.model}, Dataset: {args.dataset}')

users_model = []
if args.model == 'lenet5' and args.dataset == 'cifar10':
    net_glob = LeNet5Cifar10().to(args.device)
    net_glob.apply(weight_init)
    users_model = [LeNet5Cifar10().to(args.device).apply(weight_init) for _ in range(args.num_users)]
elif args.model == 'lenet5' and args.dataset == 'cifar100':
    net_glob = LeNet5Cifar100().to(args.device)
    net_glob.apply(weight_init)
    users_model = [LeNet5Cifar100().to(args.device).apply(weight_init) for _ in range(args.num_users)]
elif args.model == 'lenet5' and args.dataset == 'mnist':
    net_glob = LeNet5Mnist().to(args.device)
    net_glob.apply(weight_init)
    users_model = [LeNet5Mnist().to(args.device).apply(weight_init) for _ in range(args.num_users)]

if args.load_initial:
    initial_state_dict = torch.load(args.load_initial)
    net_glob.load_state_dict(initial_state_dict)

initial_state_dict = copy.deepcopy(net_glob.state_dict())
server_state_dict = copy.deepcopy(net_glob.state_dict())

for i in range(args.num_users):
    users_model[i].load_state_dict(initial_state_dict)
    
## 
mask_init = make_init_mask(net_glob)

clients = []
    
for idx in range(args.num_users):
    clients.append(Client_Sub_Un(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep, 
               args.lr, args.momentum, args.device, copy.deepcopy(mask_init), 
               args.pruning_target, train_dataset, user_groups_train[idx], 
               test_dataset, user_groups_test[idx])) 
    
## 
loss_train = []

init_tracc_pr = []  # initial train accuracy for each round 
final_tracc_pr = [] # final train accuracy for each round 

init_tacc_pr = []  # initial test accuarcy for each round 
final_tacc_pr = [] # final test accuracy for each round

init_tloss_pr = []  # initial test loss for each round 
final_tloss_pr = [] # final test loss for each round 

clients_best_acc = [0 for _ in range(args.num_users)]
w_locals, loss_locals = [], []
masks = []

init_local_tacc = []       # initial local test accuracy at each round 
final_local_tacc = []  # final local test accuracy at each round 

init_local_tloss = []      # initial local test loss at each round 
final_local_tloss = []     # final local test loss at each round 

ckp_avg_tacc = []
ckp_avg_pruning = []
ckp_avg_best_tacc_before = []
ckp_avg_best_tacc_after = []

for iteration in range(args.rounds):
        
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    
    if args.is_print:
        print(f'###### ROUND {iteration+1} ######')
        print(f'Clients {idxs_users}')
    
    for idx in idxs_users:
                    
        if iteration+1 > 1:
            dic = Sub_FedAvg_U_initial(copy.deepcopy(clients[idx].get_mask()), 
                                     copy.deepcopy(clients[idx].get_net()), server_state_dict)
            
            clients[idx].set_state_dict(dic) 
        
        loss, acc = clients[idx].eval_test()        
            
        init_local_tacc.append(acc)
        init_local_tloss.append(loss)
            
        loss = clients[idx].train(args.pruning_percent, args.dist_thresh, args.acc_thresh, is_print=False)
                        
        masks.append(copy.deepcopy(clients[idx].get_mask()))     
        w_locals.append(copy.deepcopy(clients[idx].get_state_dict()))
        loss_locals.append(copy.deepcopy(loss))
                        
        loss, acc = clients[idx].eval_test()
        
        if acc > clients_best_acc[idx]:
            clients_best_acc[idx] = acc
  
        final_local_tacc.append(acc)
        final_local_tloss.append(loss)           
        
    server_state_dict = Sub_FedAVG_U(server_state_dict, w_locals, masks)
    
    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    avg_init_tloss = sum(init_local_tloss) / len(init_local_tloss)
    avg_init_tacc = sum(init_local_tacc) / len(init_local_tacc)
    avg_final_tloss = sum(final_local_tloss) / len(final_local_tloss)
    avg_final_tacc = sum(final_local_tacc) / len(final_local_tacc)
    
    if args.is_print:    
        print('## END OF ROUND ##')
        template = 'Average Train loss {:.3f}'
        print(template.format(iteration+1, loss_avg))

        template = "AVG Init Test Loss: {:.3f}, AVG Init Test Acc: {:.3f}"
        print(template.format(avg_init_tloss, avg_init_tacc))

        template = "AVG Final Test Loss: {:.3f}, AVG Final Test Acc: {:.3f}"
        print(template.format(avg_final_tloss, avg_final_tacc))

    if iteration%args.print_freq == 0:
        print('--- PRINTING ALL CLIENTS STATUS ---')
        best_acc_before_pruning = []
        pruning_state = []
        current_acc = []
        for k in range(args.num_users):
            best_acc_before_pruning.append(clients[k].get_best_acc())
            pruning_state.append(clients[k].get_pruning())
            loss, acc = clients[k].eval_test() 
            current_acc.append(acc)
            
            template = ("Client {:3d}, labels {}, count {}, pruning_state {:3.3f}, "
                       "best_acc_befor_pruning {:3.3f}, after_pruning {:3.3f}, current_acc {:3.3f} \n")
            
            print(template.format(k, users_train_labels[k], clients[k].get_count(), pruning_state[-1], 
                                 best_acc_before_pruning[-1], clients_best_acc[k], current_acc[-1]))
            
        template = ("Round {:1d}, Avg Pruning {:3.3f}, Avg current_acc {:3.3f}, "
                     "Avg best_acc_before_pruning {:3.3f}, after_pruning {:3.3f}")
        
        print(template.format(iteration+1, np.mean(pruning_state), np.mean(current_acc), 
                              np.mean(best_acc_before_pruning), np.mean(clients_best_acc)))
        
        ckp_avg_tacc.append(np.mean(current_acc))
        ckp_avg_pruning.append(np.mean(pruning_state))
        ckp_avg_best_tacc_before.append(np.mean(best_acc_before_pruning))
        ckp_avg_best_tacc_after.append(np.mean(clients_best_acc))
        
    loss_train.append(loss_avg)
    
    init_tacc_pr.append(avg_init_tacc)
    init_tloss_pr.append(avg_init_tloss)
    
    final_tacc_pr.append(avg_final_tacc)
    final_tloss_pr.append(avg_final_tloss)
    
    ## clear the placeholders for the next round 
    masks.clear()
    w_locals.clear()
    loss_locals.clear()
    init_local_tacc.clear()
    init_local_tloss.clear()
    final_local_tacc.clear()
    final_local_tloss.clear()
    
    ## calling garbage collector 
    gc.collect()
    
## Printing Final Test and Train ACC / LOSS
test_loss = []
test_acc = []
train_loss = []
train_acc = []

for idx in range(args.num_users):        
    loss, acc = clients[idx].eval_test()
        
    test_loss.append(loss)
    test_acc.append(acc)
    
    loss, acc = clients[idx].eval_train()
    
    train_loss.append(loss)
    train_acc.append(acc)

test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(test_acc) / len(test_acc)

train_loss = sum(train_loss) / len(train_loss)
train_acc = sum(train_acc) / len(train_acc)

print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')
print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')