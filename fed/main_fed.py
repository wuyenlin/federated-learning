#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
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

#def scale(x, out_range=(-1, 1)):
#    domain = np.min(x), np.max(x)
#    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
#    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.gpu = 0
    args.iid = True
    args.dataset = 'mnist'
    args.num_channels = 1
    args.model = 'cnn'
    
    args.epochs = 100
    args.local_ep = 20
    args.local_bs = 10
    
    results_file = open("results_file_var_0dot3.txt","a")
    
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    
    # To save or not
    save_reconstructed=1
    save_original=1
    
    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            
            ################## Adding some noise            
            # The first layer of the network. loc = user, glob = global  
            #w_conv1_loc = w["conv1.weight"]
            #b_conv1_loc = w["conv1.bias"]
            #w_conv1_glob = w_glob["conv1.weight"]
            #b_conv1_glob = w_glob["conv1.bias"]

            # add noise
            for layer in w:
                x = np.random.normal(0,0.3,w[layer].size())
                x = np.reshape(x,w[layer].size())
                x = torch.from_numpy(x)
                w[layer] = w[layer]+x.cuda()
            
            ################## Create Toeplitz matrix and derive original image
            #w_conv1_glob_np = w_conv1_glob[0][0][:][:].cpu().numpy() # 5x5 weight matrix global model
            #w_conv1_loc_np = w_conv1_loc[0][0][:][:].cpu().numpy()  # 5x5 weight matrix given by client
            
            #n = 28
            #m = 5
            #remember_jump=0
            #w_global_matrix = np.zeros(((n-m+1)**2,n*n)) # Toeplitz matrix
            #w_local_matrix = np.zeros(((n-m+1)**2,n*n)) # Toeplitz matrix
            
            
            # Convert to doubly Toeplitz matrix https://github.com/alisaaalehi/convolution_as_multiplication/blob/master/Convolution_as_multiplication.ipynb
            #for i in range((n-m+1)**2):
            #    if (i%(n-m+1)==0 and i!=0):
            #        for j in range(m):
                        #print(i,'\t +2')
            #            w_global_matrix[i][n*j+i+m-1+remember_jump : n*j+i+m+m-1+remember_jump] = w_conv1_glob_np[j][:]
            #            w_local_matrix[i][n*j+i+m-1+remember_jump : n*j+i+m+m-1+remember_jump] = w_conv1_loc_np[j][:]
            #        remember_jump = remember_jump+1
            #    else:
            #        for j in range(m):
                        #print(i, '\t +1')
            #            w_global_matrix[i][n*j+i+remember_jump : n*j+i+m+remember_jump] = w_conv1_glob_np[j][:]
            #            w_local_matrix[i][n*j+i+remember_jump : n*j+i+m+remember_jump] = w_conv1_loc_np[j][:]

            #delta_w = w_global_matrix-w_local_matrix
            #delta_b = (b_conv1_glob[0]-b_conv1_loc[0]).cpu().numpy()
            #x = delta_w / delta_b
            #x = delta_w
            
            # Subtract local model weight and bias from global model weight and bias and divide
            #x = (w_conv1_glob[0][0][:][:]-w_conv1_loc[0][0][:][:]) / (b_conv1_glob[0]-b_conv1_loc[0])
            #x=scale(x.cpu().numpy(),(0,1))
            
            #x_aggr=np.zeros((np.shape(x)[0],5*5))
            #for k in range(np.shape(x)[0]): # for every row get the digits that are /= 0.... only first
            #    if (k%24==0 and k>0):
            #        x_aggr[k][0:5] = x[k][0+k+4:5+k+4]
            #        x_aggr[k][5:10] = x[k][28+k+4:33+k+4]
            #        x_aggr[k][10:15] = x[k][56+k+4:61+k+4]
            #        x_aggr[k][15:20] = x[k][84+k+4:89+k+4]
            #        x_aggr[k][20:25] = x[k][112+k+4:117+k+4]
            #    else:
            #        x_aggr[k][0:5] = x[k][0+k:5+k]
            #        x_aggr[k][5:10] = x[k][28+k:33+k]
            #        x_aggr[k][10:15] = x[k][56+k:61+k]
            #        x_aggr[k][15:20] = x[k][84+k:89+k]
            #        x_aggr[k][20:25] = x[k][112+k:117+k]
            
            #print(np.shape(x), '\t', type(x))
            #if (save_reconstructed):
                #x = np.reshape(x,(10,25))
            #    x_img=x_aggr[0][:]
            #    x_img = np.reshape(x_img,(5,5))
                #x_img = x[:][0]
                #x_img = np.reshape(x_img,(28,28))
            #    plt.imshow(x_img,cmap='gray')
            #    plt.savefig("results/fig2.png")
            #    save_reconstruced=0

            #if (save_original):
            #    train_image = dataset_train[0][0]
            #    train_image = train_image[0][:][:].cpu().numpy()
            #    train_image = scale(train_image, (0,1))
            #    plt.imshow(train_image,cmap='gray')
            #    plt.savefig("results/fig_original.png")
            
            #####################################################
            
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
        # Evaluate score
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        results_file.write("%i \t %f \n" %(iter, acc_test))
    

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    
    results_file.close()    
