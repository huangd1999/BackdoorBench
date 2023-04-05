import random
import sys, argparse, yaml
import numpy as np

# print(sys.argv[0])  # print which file is the main script
sys.path.append('../')

def choose_index(args, data_all_length) :
    # choose clean data according to index
    if args.index == None:
        ran_idx = random.sample(range(data_all_length),int(data_all_length*args.ratio))
    else:
        ran_idx = np.loadtxt(args.index, dtype=int)
       
    return ran_idx


def choose_index_balanced(args, data_all_length, y, num_classes=10):
    per_class_samples_num = int(data_all_length * args.ratio // num_classes)
    idx = [np.where(y == label)[0] for label in range(num_classes)]
    for i in range(len(idx)):
        idx[i] = np.random.choice(idx[i], size=per_class_samples_num, replace=False)
    ran_idx = np.concatenate(idx, axis=-1)
    return ran_idx
