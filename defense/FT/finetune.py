import argparse
import logging
import os
import random
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

#from utils import args
from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
#from utils.input_aware_utils import progress_bar
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils.save_load_attack import load_attack_result
import yaml
from pprint import pprint, pformat

def get_args():
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

    ####添加额外
    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--result_file', type=str, help='the location of result')

    ###ft
    parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')

    arg = parser.parse_args()

    print(arg)
    return arg

def fine_tuning(arg, model, optimizer, scheduler, criterion, epoch, trainloader, testloader_cl = None, testloader_bd = None):
    model.train()

    total_clean, total_clean_correct, train_loss = 0, 0, 0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
        #progress_bar(i, len(trainloader), 'Epoch: %d | Loss: %.3f | Training Acc: %.3f%% (%d/%d)' % (epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))
        print('Epoch:{} | Loss: {:.3f} | Training Acc: {:.3f}%({}/{})'.format(epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))
        logging.info('Epoch:{} | Loss: {:.3f} | Training Acc: {:.3f}%({}/{})'.format(epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))
        if testloader_cl is not None:
            total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
            for i, (inputs, labels) in enumerate(testloader_cl):
                inputs, labels = inputs.to(arg.device), labels.to(arg.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
                total_clean_test += inputs.shape[0]
                avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
                #progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
            print('Epoch:{} | Test Acc: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct, total_clean))
            logging.info('Epoch:{} | Test Acc: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct, total_clean))
                    
        if testloader_bd is not None:
            total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
            for i, (inputs, labels) in enumerate(testloader_bd):
                inputs, labels = inputs.to(arg.device), labels.to(arg.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
                total_clean_test += inputs.shape[0]
                avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
                #progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
            print('Epoch:{} | Test Asr: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct, total_clean))
            logging.info('Epoch:{} | Test Asr: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct, total_clean))
    scheduler.step()
    return model

def train_epoch(arg, trainloader, model, optimizer, scheduler, criterion, epoch):
    model.train()

    total_clean, total_clean_correct, train_loss = 0, 0, 0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
        logging.info('Epoch[{0}]:[{1:03}/{2:03}]'
                'Loss:{losses:.4f}({losses_avg:.4f})'.format(epoch, i, len(trainloader), losses=train_loss, losses_avg=train_loss/total_clean))

    scheduler.step()
    return train_loss / (i + 1), avg_acc_clean


def test_epoch(arg, testloader, model, criterion, epoch, word):
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
        logging.info('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(avg_acc_clean, test_loss/total_clean))
    if word == 'clean':
        logging.info('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(avg_acc_clean, test_loss/total_clean))
    
    return test_loss / (i + 1), avg_acc_clean

def ft(args,result,config):
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    # logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
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

    # Prepare model, optimizer, scheduler
    model = generate_cls_model(args.model,args.num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    x = torch.tensor(nCHW_to_nHWC(result['clean_train']['x'].numpy()))
    y = result['clean_train']['y']
    data_all_length = y.size()[0]
    ran_idx = random.sample(range(data_all_length),int(data_all_length*args.ratio))
    data_set = torch.utils.data.TensorDataset(x[ran_idx],y[ran_idx])
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
    model_new = fine_tuning(args, model, optimizer, scheduler, criterion, args.epoch, trainloader, testloader_cl = None, testloader_bd = None)
    
    result = {}
    result['model'] = model_new
    return result


# def main():
#     global arg
#     arg = args.get_args()

#     # Dataset
#     trainloader = get_dataloader(arg, True)
#     testloader_clean, testloader_bd = get_dataloader_test(arg)

#     # Prepare model, optimizer, scheduler
#     model = get_network(arg)
#     optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

#     checkpoint = torch.load(arg.checkpoint_load)
#     print("Start finetuning model")
#     model.load_state_dict(checkpoint['model'])
#     start_epoch = 0

#     # Training and Testing
#     best_acc = 0
#     criterion = nn.CrossEntropyLoss()

#     for epoch in tqdm(range(start_epoch, arg.epochs)):
#         train_loss, train_acc = train_epoch(arg, trainloader, model, optimizer, scheduler, criterion, epoch)
#         test_loss, test_acc_cl = test_epoch(arg, testloader_clean, model, criterion, epoch, 'clean')
#         test_loss, test_acc_bd = test_epoch(arg, testloader_bd, model, criterion, epoch, 'bd')

#         if test_acc_cl > best_acc:
#             best_acc = test_acc_cl
#             save_checkpoint(arg.checkpoint_save, epoch, model, optimizer, scheduler)


if __name__ == '__main__':
    args = get_args()
    with open("./defense/FT/config/config.yaml", 'r') as stream: 
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
    
    

    ######为了测试临时写的代码
    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/record/defence/ft/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/ft/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log)  
    args.save_path = save_path
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    if args.save_path is not None:
        print("Continue training...")
        result_defense = ft(args,result,config)

        tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].numpy()))
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
        x = torch.tensor(nCHW_to_nHWC(result['clean_test']['x'].numpy()))
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

        if not (os.path.exists(os.getcwd() + f'{save_path}/ft/')):
            os.makedirs(os.getcwd() + f'{save_path}/ft/')
        torch.save(
        {
            'model_name':args.model,
            'model': result_defense['model'].cpu().state_dict(),
            'asr': asr_acc,
            'acc': clean_acc
        },
        os.getcwd() + f'{save_path}/ft/defense_result.pt'
        )
    else:
        print("There is no target model")
