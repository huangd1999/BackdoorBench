from ast import arg
import logging
import time
import torch.nn.functional as F
from calendar import c
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, GTSRB, ImageFolder
import torch
import logging
import argparse
import sys
from tqdm import tqdm
import torch.nn as nn
import os
from torch.utils.data import random_split
sys.path.append('../')
sys.path.append(os.getcwd())

import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform, get_dataset_mean_std
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import *
from utils.save_load_attack import load_attack_result
sys.path.append(os.getcwd())
import yaml
from pprint import pprint, pformat


import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from utils.choose_index import choose_index_balanced

import matplotlib.pyplot as plt
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, x):
        return (x - self.mean) / self.std

def dynamiccluster(arrays):
    score_list = list()
    arrays = np.array(arrays)
    plt.scatter(arrays[:,0], arrays[:,2], s=10)
    acc_diff_1 = arrays[:,1]-arrays[:,0]
    acc_diff_2 = arrays[:,2]-arrays[:,1]
    diff_arrays = np.concatenate((acc_diff_1[:,None], acc_diff_2[:,None]), axis=1)
    arrays = diff_arrays            # using acc change to cluster
    # print('clustering...')
    # print(arrays.shape)
    plt.scatter(diff_arrays[:,0], diff_arrays[:,1], s=10)
    
    plt.savefig('./diff_array.png')
    print('saved')
    silhouette_int = float("-inf")
    for n_clusters in range(2, 10):
        model_kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels_tmp =model_kmeans.fit_predict(arrays)
        silhouette_tmp =metrics.silhouette_score(arrays, cluster_labels_tmp)
        if silhouette_tmp >silhouette_int:
            best_k =n_clusters
            silhouette_int =silhouette_tmp
            best_kmeans =model_kmeans
            cluster_labels_k =cluster_labels_tmp
    score_list.append([n_clusters, silhouette_tmp])
    minor =  np.argmin(np.bincount(cluster_labels_k))
    # print(len([idx for idx in range(len(cluster_labels_k)) if cluster_labels_k[idx] == minor]))
    return [idx for idx in range(len(cluster_labels_k)) if cluster_labels_k[idx] == minor]

def anomaly_det(arrays):
    score_list = list()
    arrays = np.array(arrays)
    # plt.scatter(arrays[:,0], arrays[:,2], s=10)
    acc_diff_1 = arrays[:,1]
    acc_diff_2 = arrays[:,0]
    diff_arrays = np.concatenate((acc_diff_1[:,None], acc_diff_2[:,None]), axis=1)
    arrays = diff_arrays            # using acc change to cluster
    # print('clustering...')
    # print(arrays.shape)
    # print(arrays)
    # plt.scatter(diff_arrays[:,0], diff_arrays[:,1], s=10)
    algo = EllipticEnvelope(contamination = 0.1, support_fraction=1.)
    # algo = svm.OneClassSVM(nu = 0.1, kernel='rbf', gamma=0.1)
    

    y_pred = algo.fit(arrays).predict(arrays)
    # print(y_pred)
    idx = np.where(y_pred == -1)[0]
    
    # plt.savefig('./diff_array.png')
    # print('saved')
    return idx


class Norm_layer(nn.Module):
    def __init__(self,mean,std) -> None:
        super(Norm_layer,self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1),requires_grad = False)

        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1),requires_grad = False)

    def forward(self,x):
        return x.sub(self.mean).div(self.std)


def get_args():
    #set the basic parameter
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str)
    parser.add_argument('--checkpoint_save', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)
    parser.add_argument("--inference", type=bool, default = False)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--num_workers", type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')

    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
  
    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--random_seed', type=int, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--result_file', type=str, help='the location of result')

    parser.add_argument('--yaml_path', type=str, default="./config/defense/ac/config.yaml", help='the path of yaml')

    #set the parameter for the ac defense
    parser.add_argument('--nb_dims', type=int, help='umber of dimensions to reduce activation to')
    parser.add_argument('--nb_clusters', type=int, help='number of clusters (defaults to 2 for poison/clean).')
    parser.add_argument('--cluster_analysis', type=str, help='the method of cluster analysis')
    
    arg = parser.parse_args()

    print(arg)
    return arg

def ranking(model,adv_list,label):
    rank_list = []
    for data in adv_list:
        result_list = []
        for image in data:
            pred = model(image.cuda())
            pred = np.argmax(pred.cpu().detach(), axis=-1)
            correct = pred == label
            correct = np.sum(correct.numpy(), axis=-1)
            result_list.append(correct/image.shape[0])

        rank_list.append(result_list)
    return rank_list

def obtain_adv_dataset(model,splitdata):
    mean, std = get_dataset_mean_std(args.dataset)
    if args.dataset == 'tiny':
        dataset = ImageFolder('./data/tiny/tiny-imagenet-200/train/', transform=ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        model = nn.Sequential(Norm_layer(mean,std),model)
        model = model.cuda()
    elif args.dataset == 'cifar10':
        dataset = CIFAR10('./data/cifar10', train=True, download=True, transform=ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=128)
        model = nn.Sequential(Norm_layer(mean,std),model)
        model = model.cuda()
    elif args.dataset == 'cifar100':
        dataset = CIFAR100('./data/cifar100', train=True, download=True, transform=ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        model = nn.Sequential(Norm_layer(mean,std),model)
        model = model.cuda()
    elif args.dataset == 'gtsrb':
        dataset = GTSRB('./data/gtsrb', train=True, download=True, transform=ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        model = nn.Sequential(Norm_layer(mean,std),model)
        model = model.cuda()

    for idx, (data, label) in enumerate(tqdm(train_loader)):
        x = data
        y = label
        adv_images = x + args.alpha*torch.empty_like(x).uniform_(-args.alpha, args.alpha).sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach().cuda()
        break
    conv_outputs = []
    def get_conv_output_hook(module, input, output):
        if isinstance(module, nn.Conv2d):
            conv_outputs.append(output)

    # Register the hook on all convolutional layers
    def model_hook(model):
        handles = []
        for module in model.modules():
            handle = module.register_forward_hook(get_conv_output_hook)
            handles.append(handle)
        return handles

    def remove_hook(handles):
        for handle in handles:
            handle.remove
    handles = model_hook(model)
    x = x.cuda()
    output = model(x)
    num_conv_layer = len(conv_outputs)
    total_feature = 0
    inter_node = [0]
    for layer in conv_outputs:
        total_feature+=layer.shape[1]
        inter_node.append(total_feature)
    for handle in handles:
        handle.remove()
    
    def adv_sample_generation(idx,x, adv_images):
        image_list = []
        for each_adv_idx in range(args.steps):
            conv_outputs = []
            def get_conv_output_hook(module, input, output):
                if isinstance(module, nn.Conv2d):
                    conv_outputs.append(output)

            # Register the hook on all convolutional layers
            def model_hook(model):
                handles = []
                for module in model.modules():
                    handle = module.register_forward_hook(get_conv_output_hook)
                    handles.append(handle)
                return handles

            def remove_hook(handles):
                for handle in handles:
                    handle.remove
            handles = model_hook(model)
            x = x.cuda()
            adv_images = adv_images.requires_grad_(True).cuda()
            output = model(x)
            adv_output = model(adv_images)
            clean_feature_maps = []
            adv_feature_maps = []
            start = 0
            for i, conv_output in enumerate(conv_outputs):
                if i<num_conv_layer:
                    if start<=idx and start+conv_output.shape[1]>idx:
                        clean_feature_map = conv_output[:,idx-start,:,:]
                    start = start+conv_output.shape[1]
                    continue

                if i==num_conv_layer:
                    start = 0
                    if start<=idx and start+conv_output.shape[1]>idx:
                        adv_feature_map = conv_output[:,idx-start,:,:]
                    start = start+conv_output.shape[1]
                    continue
                if i>num_conv_layer:
                    if start<=idx and start+conv_output.shape[1]>idx:
                        adv_feature_map = conv_output[:,idx-start,:,:]
                    start = start+conv_output.shape[1]

            loss = F.mse_loss(clean_feature_map,adv_feature_map)
            grad = torch.autograd.grad(loss, adv_images,
                                        retain_graph=True, create_graph=False)[0]

            adv_images = adv_images.detach() + args.alpha*grad.sign()
            delta = torch.clamp(adv_images - x, min=-args.eps, max=args.eps)
            adv_images = torch.clamp(x + delta, min=0, max=1).detach()
            for handle in handles:
                handle.remove()
            image_list.append(adv_images.cpu())
        return image_list

    adv_list = []

    rank_list = []

    layer_idx = 1
    for i in tqdm(range(total_feature)):
        if i<inter_node[layer_idx]:
            adv_list.append(adv_sample_generation(i, x, adv_images))
        else:
            layer_idx+=1
            rank_list.append(anomaly_det(ranking(model,adv_list,y)))
            adv_list = []
            adv_list.append(adv_sample_generation(i, x, adv_images))
    
    rank_list.append(anomaly_det(ranking(model,adv_list,y)))


    return rank_list

def train(model,train_loader,test_loader,rank_list):
    total = 0
    clean_acc = 0
    for i, (inputs,labels) in enumerate(test_loader):
        inputs = inputs.to(args.device)
        outputs = model(inputs)
        pred = np.argmax(outputs.cpu().detach(), axis=-1)
        curr_correct = pred == labels
        clean_acc += np.sum(curr_correct.numpy(), axis=-1)
        total+=len(labels)
    print('epoch: {} test acc: {}'.format(0, clean_acc/total))
    print('length rank_list', len(rank_list))
    start_idx = 0
    for parameter in model.parameters():
        # parameter.requires_grad = False
        # print(parameter.data.shape)
        if len(parameter.data.shape)==4:
            start_idx+=1
            # if start_idx != 20:
            #     continue
            helper = rank_list[start_idx-1]
            # print(helper)
            # np.random.seed(1)
            # if len(helper) == 7:
            #     helper = np.random.choice(np.linspace(0,63,63, dtype=int), 7, replace=False)
            # if len(helper) == 13:
            #     helper = np.random.choice(np.linspace(0,127,127, dtype=int), 13, replace=False)
            # if len(helper) == 26:
            #     helper = np.random.choice(np.linspace(0,255,255, dtype=int), 26, replace=False)
            # if len(helper) == 52:
            #     helper = np.random.choice(np.linspace(0,511,511, dtype=int), 52, replace=False)
            # print(helper)
            # print(parameter.data.shape)
            for fmidx in helper:
                # print(fmidx)
                # print(parameter.data[fmidx])
                torch.nn.init.zeros_(parameter.data[fmidx])
                # torch.nn.init.kaiming_uniform_(parameter.data[fmidx])
                # print(parameter.data[fmidx])
            # parameter.requires_grad = True
            

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        for i, (inputs,labels) in enumerate(tqdm(train_loader)):
            model.train()
            model.to(args.device)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # scheduler.step()
        total = 0
        clean_acc = 0
        for i, (inputs,labels) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            pred = np.argmax(outputs.cpu().detach(), axis=-1)
            curr_correct = pred == labels
            clean_acc += np.sum(curr_correct.numpy(), axis=-1)
            total+=len(labels)
        print('epoch: {} test acc: {}'.format(epoch, clean_acc/total))
    return model

if __name__ == "__main__":
    ### 1. basic setting: args
    args = get_args()
    with open(args.yaml_path, 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)

    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/record/defence/feature/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/feature/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log) 
    args.save_path = save_path

    ### 2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    model = generate_cls_model(args.model,args.num_classes)
    model.load_state_dict(result['model'])
    model = model.to(args.device)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x,y))
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x,y))
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    x = result['clean_train']['x']
    y = result['clean_train']['y']
    data_all_length = len(y)
    ran_idx = choose_index_balanced(args, data_all_length, y) 
    log_index = os.getcwd() + args.log + 'index.txt'
    np.savetxt(log_index,ran_idx, fmt='%d')
    data_clean_train = list(zip([x[ii] for ii in ran_idx],[y[ii] for ii in ran_idx]))
    data_clean_trainset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_train,
        poison_idx=np.zeros(len(data_clean_train)),
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_train_loader = torch.utils.data.DataLoader(data_clean_trainset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)


    if args.inference:
        helper = torch.load(os.getcwd() + f'{save_path}/feature/defense_result.pt')
        model.load_state_dict(helper['model'])
        print('clean acc: {}'.format(helper['clean_acc']))
        print('ASR acc: {}'.format(helper['asr_acc']))
    else:
        total_bd = 0
        total_clean = 0
        total_train = 0
        with torch.no_grad():
            model.eval()
            asr_acc = 0
            for i, (inputs,labels) in enumerate(data_bd_loader):
                inputs = inputs.to(args.device)
                outputs = model(inputs)
                pred = np.argmax(outputs.cpu().detach(), axis=-1)
                curr_correct = pred == labels
                asr_acc += np.sum(curr_correct.numpy(), axis=-1)
                total_bd += len(labels)
            clean_acc = 0
            for i, (inputs,labels) in enumerate(data_clean_loader):
                inputs = inputs.to(args.device)
                outputs = model(inputs)
                pred = np.argmax(outputs.cpu().detach(), axis=-1)
                curr_correct = pred == labels
                clean_acc += np.sum(curr_correct.numpy(), axis=-1)
                total_clean += len(labels)
            clean_correct = 0
            for i, (inputs,labels) in enumerate(data_train_loader):

                inputs = inputs.to(args.device)
                outputs = model(inputs)
                pred = np.argmax(outputs.cpu().detach(), axis=-1)
                curr_correct = pred == labels
                clean_correct += np.sum(curr_correct.numpy(), axis=-1)
                total_train += len(labels)
            print(asr_acc, asr_acc/total_bd,  clean_acc, clean_acc/total_clean, clean_correct, clean_correct/total_train)
        rank_list = obtain_adv_dataset(model,data_clean_trainset)
        # rank_list = torch.load(os.getcwd() + f'{save_path}/feature/empirical_cov_partition0.1_acc_defense_data0.05_result.pt')['rank_list']
        model = train(model,data_train_loader,data_clean_loader,rank_list)
        with torch.no_grad():
            model.eval()
            asr_acc = 0
            for i, (inputs,labels) in enumerate(data_bd_loader):
                inputs = inputs.to(args.device)
                outputs = model(inputs)
                pred = np.argmax(outputs.cpu().detach(), axis=-1)
                curr_correct = pred == labels
                asr_acc += np.sum(curr_correct.numpy(), axis=-1)
            clean_correct = 0
            for i, (inputs,labels) in enumerate(data_clean_loader):

                inputs = inputs.to(args.device)
                outputs = model(inputs)
                pred = np.argmax(outputs.cpu().detach(), axis=-1)
                curr_correct = pred == labels
                clean_correct += np.sum(curr_correct.numpy(), axis=-1)
            print(asr_acc/total_bd, clean_correct/total_clean)
        if not (os.path.exists(os.getcwd() + f'{save_path}/feature/')):
            os.makedirs(os.getcwd() + f'{save_path}/feature/')
        torch.save(
        {
            'model_name':args.model,
            'model':model.state_dict(),
            'clean_acc':clean_acc/total_clean,
            'asr_acc':asr_acc/total_bd,
            'rank_list':rank_list
        }, os.getcwd() + f'{save_path}/feature/empirical_cov_partition0.1_acc_defense_data0.05_result.pt')
    
    with open(os.getcwd() + f'{save_path}/feature/defense_result.txt', 'w') as f:
        f.write('asr: '+str(asr_acc)+' acc: '+str(acc))
        f.close()