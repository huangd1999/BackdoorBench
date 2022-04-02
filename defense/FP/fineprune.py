'''
This file is modified based on the following source:
link : https://github.com/kangliucn/Fine-pruning-defense
The defense method is called fp.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. add some addtional backbone such as resnet18 and vgg19
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. fp defense:
        a. hook the activation layer representation of each data
        b. rank the mean of activation for each neural
        c. according to the sorting results, prune and test the accuracy
        d. save the model with the greatest difference between ACC and ASR
    4. test the result and get ASR, ACC, RC 
'''


import argparse
import logging
import os
import sys 
sys.path.append('../')
sys.path.append(os.getcwd())
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import numpy as np

from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils.save_load_attack import load_attack_result
import yaml
from pprint import pprint, pformat

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

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--num_workers", type=float)
    parser.add_argument('--lr', type=float)

    parser.add_argument('--attack', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--result_file', type=str, help='the location of result')

    arg = parser.parse_args()

    print(arg)
    return arg


def test_epoch(arg, testloader, model, criterion, epoch, word):
    '''test the student model with regard to test data for each epoch
    arg:
        Contains default parameters
    testloader:
        the dataloader of clean test data or backdoor test data
    model:
        the training model
    criterion:
        criterion during the train process
    epoch:
        current epoch
    word:
        'bd' or 'clean'
    '''
    model.eval()

    total_clean, total_clean_correct, test_loss = 0, 0, 0

    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)

        if word == 'bd':
            logging.info(f'test_Epoch{i}: asr:{avg_acc_clean}({total_clean_correct}/{total_clean})')
            #progress_bar(i, len(testloader), 'Test %s ASR: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
        if word == 'clean':
            logging.info(f'test_Epoch{i}: clean_acc:{avg_acc_clean}({total_clean_correct}/{total_clean})')
            #progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))

    return test_loss / (i + 1), avg_acc_clean

def fp(args, result , config):
    ### set logger
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    if args.log is not None and args.log != '':
        fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    else:
        fileHandler = logging.FileHandler(os.getcwd() + './log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    ### a. hook the activation layer representation of each data
    # Prepare model
    netC = generate_cls_model(args.model,args.num_classes)
    netC.load_state_dict(result['model'])
    netC.to(args.device)
    netC.eval()
    netC.requires_grad_(False)

    criterion = nn.CrossEntropyLoss()
    # Prepare dataloader and check initial acc_clean and acc_bd
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    x = torch.tensor(nCHW_to_nHWC(result['bd_train']['x'].detach().numpy()))
    y = result['bd_train']['y']
    data_set = torch.utils.data.TensorDataset(x,y)
    data_set_o = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    trainloader = data_loader
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].detach().numpy()))
    y = result['bd_test']['y']
    data_bd_test = torch.utils.data.TensorDataset(x,y)
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = torch.tensor(nCHW_to_nHWC(result['clean_test']['x'].detach().numpy()))
    y = result['clean_test']['y']
    data_clean_test = torch.utils.data.TensorDataset(x,y)
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    testloader_bd = data_bd_loader
    testloader_clean = data_clean_loader
    for name, module in netC._modules.items():
        print(name)

    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    if args.model == 'preactresnet18':
        hook = netC.layer4.register_forward_hook(forward_hook)
    if args.model == 'vgg19':
        hook = netC.features.register_forward_hook(forward_hook)
    if args.model == 'resnet18':
        hook = netC.layer4.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(args.device)
        netC(inputs)
    hook.remove()

    ### b. rank the mean of activation for each neural
    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
  
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    if args.model == 'preactresnet18':
        addtional_dim = 1
        pruning_mask_li = torch.ones(pruning_mask.shape[0] * addtional_dim, dtype=bool)
    if args.model == 'vgg19':
        addtional_dim = 49
        pruning_mask_li = torch.ones(pruning_mask.shape[0] * addtional_dim, dtype=bool)
    if args.model == 'resnet18':
        addtional_dim = 1
        pruning_mask_li = torch.ones(pruning_mask.shape[0] * addtional_dim, dtype=bool)
    
    ### c. according to the sorting results, prune and test the accuracy
    acc_dis = 0
    # Pruning times - no-tuning after pruning a channel!!!
    for index in range(int(pruning_mask.shape[0])):
        net_pruned = copy.deepcopy(netC)
        num_pruned = index
        if index:
            channel = seq_sort[index - 1]
            pruning_mask[channel] = False
            pruning_mask_li[range(channel*addtional_dim,((channel+1)*addtional_dim))] = False
        print("Pruned {} filters".format(num_pruned))
        if args.model == 'preactresnet18':
            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.linear = nn.Linear((pruning_mask.shape[0] - num_pruned)*addtional_dim, 10)
        if args.model == 'vgg19':
            net_pruned.features[34] = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.classifier[0] = nn.Linear((pruning_mask.shape[0] - num_pruned)*addtional_dim, 4096)
        if args.model == 'resnet18':
            net_pruned.layer4[0].conv1 = nn.Conv2d(
                256, pruning_mask.shape[0] - num_pruned, (3, 3), stride=(2, 2), padding=1, bias=False
            )
            net_pruned.layer4[0].bn1 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            net_pruned.layer4[0].conv2 = nn.Conv2d(
                pruning_mask.shape[0] - num_pruned, pruning_mask.shape[0] - num_pruned, (3, 3), stride=(1, 1), padding=1, bias=False
            )
            net_pruned.layer4[0].bn2 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            net_pruned.layer4[0].downsample[0] = nn.Conv2d(
                256, pruning_mask.shape[0] - num_pruned, (1, 1), stride=(2, 2), bias=False
            )
            net_pruned.layer4[0].downsample[1] = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            
            net_pruned.layer4[1].conv1 = nn.Conv2d(
                pruning_mask.shape[0] - num_pruned, pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.layer4[1].bn1 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0] - num_pruned, pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.layer4[1].bn2 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
            net_pruned.fc = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)

        # Re-assigning weight to the pruned net
        for name, module in net_pruned._modules.items():
            if args.model == 'preactresnet18':
                if "layer4" == name:
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                    module[1].ind = pruning_mask
                elif "linear" == name:
                    module.weight.data = netC.linear.weight.data[:, pruning_mask_li]
                    module.bias.data = netC.linear.bias.data
                else:
                    continue
            if args.model == 'vgg19':
                if "features" == name:  
                    module[34].weight.data = netC.features[34].weight.data[pruning_mask]
                    module[34].ind = pruning_mask
                elif "classifier" == name:
                    module[0].weight.data = netC.classifier[0].weight.data[:, pruning_mask_li]
                    module[0].bias.data = netC.classifier[0].bias.data
                else:
                    continue
            if args.model == 'resnet18':
                if "layer4" == name:
                    module[0].conv1.weight.data = netC.layer4[0].conv1.weight.data[pruning_mask]
                    module[0].bn1.weight.data = netC.layer4[0].bn1.weight.data[pruning_mask]
                    module[0].bn1.bias.data = netC.layer4[0].bn1.bias.data[pruning_mask]
                    module[0].conv2.weight.data = netC.layer4[0].conv2.weight.data[pruning_mask][:,pruning_mask]
                    module[0].bn2.weight.data = netC.layer4[0].bn2.weight.data[pruning_mask]
                    module[0].bn2.bias.data = netC.layer4[0].bn2.bias.data[pruning_mask]
                    module[0].downsample[0].weight.data = netC.layer4[0].downsample[0].weight.data[pruning_mask]
                    module[0].downsample[1].weight.data = netC.layer4[0].downsample[1].weight.data[pruning_mask]
                    module[0].downsample[1].bias.data = netC.layer4[0].downsample[1].bias.data[pruning_mask]

                    module[1].conv1.weight.data = netC.layer4[1].conv1.weight.data[pruning_mask][:,pruning_mask]
                    module[1].bn1.weight.data = netC.layer4[1].bn1.weight.data[pruning_mask]
                    module[1].bn1.bias.data = netC.layer4[1].bn1.bias.data[pruning_mask]
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask][:,pruning_mask]
                    module[1].bn2.weight.data = netC.layer4[1].bn2.weight.data[pruning_mask]
                    module[1].bn2.bias.data = netC.layer4[1].bn2.bias.data[pruning_mask]
                    module[1].ind = pruning_mask
                elif "fc" == name:
                    module.weight.data = netC.fc.weight.data[:, pruning_mask_li]
                    module.bias.data = netC.fc.bias.data
                else:
                    continue
        net_pruned.to(args.device)
        test_loss, test_acc_cl = test_epoch(args, testloader_clean, net_pruned, criterion, 0, 'clean')
        test_loss, test_acc_bd = test_epoch(args, testloader_bd, net_pruned, criterion, 0, 'bd')
        print('Acc Clean: {:.3f} | Acc Bd: {:.3f}'.format(test_acc_cl, test_acc_bd)) 
        logging.info("%d %0.4f %0.4f\n" % (index, test_acc_cl, test_acc_bd))
        ### d. save the model with the greatest difference between ACC and ASR
        if index == 0:
            acc_dis = test_acc_cl - test_acc_bd
            best_net = copy.deepcopy(net_pruned)
        if test_acc_cl - test_acc_bd > acc_dis:
            best_net = copy.deepcopy(net_pruned)
            acc_dis = test_acc_cl - test_acc_bd
        if args.device == 'cuda':
            net_pruned.to('cpu')
        del net_pruned

    result = {}
    result['model'] = best_net
    return result

if __name__ == '__main__':
    
    ### 1. basic setting: args
    args = get_args()
    with open("./defense/FP/config.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    if args.dataset == "mnist":
        args.num_classes = 10
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "celeba":
        args.num_classes = 8
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    elif args.dataset == "tiny":
        args.num_classes = 200
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/record/defence/fp/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/fp/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log)  
    args.save_path = save_path

    ### 2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    ### 3. fp defense:
    result_defense = fp(args,result,config)

    ### 4. test the result and get ASR, ACC, RC 
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].detach().numpy()))
    y = result['bd_test']['y']
    data_bd_test = torch.utils.data.TensorDataset(x,y)
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    asr_acc = 0
    for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs,dim=1)[1]
        asr_acc += torch.sum(pre_label == labels)/len(data_bd_test)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = torch.tensor(nCHW_to_nHWC(result['clean_test']['x'].detach().numpy()))
    y = result['clean_test']['y']
    data_clean_test = torch.utils.data.TensorDataset(x,y)
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    clean_acc = 0
    for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs,dim=1)[1]
        clean_acc += torch.sum(pre_label == labels)/len(data_clean_test)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].detach().numpy()))
    robust_acc = -1
    if 'original_targets' in result['bd_test']:
        y_ori = result['bd_test']['original_targets']
        if y_ori is not None:
            if len(y_ori) != x.size(0):
                y_idx = result['bd_test']['original_index']
                y = y_ori[y_idx]
            else :
                y = y_ori
            data_bd_test = torch.utils.data.TensorDataset(x,y)
            data_bd_testset = prepro_cls_DatasetBD(
                full_dataset_without_transform=data_bd_test,
                poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
                bd_image_pre_transform=None,
                bd_label_pre_transform=None,
                ori_image_transform_in_loading=tran,
                ori_label_transform_in_loading=None,
                add_details_in_preprocess=False,
            )
            data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
        
            robust_acc = 0
            for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = result_defense['model'](inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                robust_acc += torch.sum(pre_label == labels)/len(data_bd_test)
        else :
            ori_label_un = result['clean_test']['y']
            ori_label = [i for i in ori_label_un if i != 0]
            y = torch.tensor(ori_label)
            data_bd_test = torch.utils.data.TensorDataset(x,y)
            data_bd_testset = prepro_cls_DatasetBD(
                full_dataset_without_transform=data_bd_test,
                poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
                bd_image_pre_transform=None,
                bd_label_pre_transform=None,
                ori_image_transform_in_loading=tran,
                ori_label_transform_in_loading=None,
                add_details_in_preprocess=False,
            )
            data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

            robust_acc = 0
            for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = result_defense['model'](inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                robust_acc += torch.sum(pre_label == labels)/len(data_bd_test)
    else:
        ori_label_un = result['clean_test']['y']
        ori_label = [i for i in ori_label_un if i != 0]
        y = torch.tensor(ori_label)
        data_bd_test = torch.utils.data.TensorDataset(x,y)
        data_bd_testset = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_bd_test,
            poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=tran,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    
        robust_acc = 0
        for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = result_defense['model'](inputs)
            pre_label = torch.max(outputs,dim=1)[1]
            robust_acc += torch.sum(pre_label == labels)/len(data_bd_test)

    if not (os.path.exists(os.getcwd() + f'{save_path}/fp/')):
        os.makedirs(os.getcwd() + f'{save_path}/fp/')
    torch.save(
    {
        'model_name':args.model,
        'model': result_defense['model'].cpu().state_dict(),
        'asr': asr_acc,
        'acc': clean_acc,
        'rc': robust_acc
    },
    os.getcwd() + f'{save_path}/fp/defense_result.pt'
    )
    