import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.gpu = -1
    args.iid = True
    args.dataset = 'mnist'
    args.num_channels = 1
    args.model = 'cnn'


    Epochs = [5] # <- change this
    B_bs = [10] # <- change this

    acc_dic = {}
    final_acc = []

    loss_dic = {}
    final_loss = []

    for i in range(len(Epochs)):
        args.epochs = Epochs[i]
        args.local_bs = B_bs[i]
        
        # load dataset and split users
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        dict_users = mnist_iid(dataset_train, args.num_users)
        img_size = dataset_train[0][0].shape
        net_glob = CNNMnist(args=args).to(args.device)

        # copy weights
        w_glob = net_glob.state_dict()

        # training
        loss_train = []
        loss_sample = []
        cv_loss, cv_acc = [], []
        val_loss_pre, counter = 0, 0
        net_best = None
        best_loss = None
        val_acc_list, net_list = [], []      

        acc_train = None 
        loss_sample = None
        acc_test = None
        loss_test = None

        for communicate in range(1,101):    
            # print("i= {}".format(i))
            # print('Round %d' % communicate)

            for iter in range(args.epochs):
                w_locals, loss_locals = [], []
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                for idx in idxs_users:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    w_locals.append(copy.deepcopy(w))
                    loss_locals.append(copy.deepcopy(loss))
                # update global weights
                w_glob = FedAvg(w_locals)

                # copy weight to net_glob
                net_glob.load_state_dict(w_glob)

                # print loss
                loss_avg = sum(loss_locals) / len(loss_locals)
                #print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
                loss_train.append(loss_avg)
            
            loss_sample = loss_train
            net_glob.eval()
            #acc_train, loss_sample = test_img(net_glob, dataset_train, args)
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            
            print(acc_test.numpy())

