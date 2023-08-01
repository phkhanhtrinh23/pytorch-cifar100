#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader

import os
import logging
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

from utils import network_to_half

start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-log', type=str, default="./logs/test_{datetime}.log", help='log file to save the logging info')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )
    
    if os.path.exists("logs/") is False:
        os.makedirs(args.log)
    
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.INFO
    log_file = args.log

    logging.basicConfig(level=log_level, format=log_format,
                        filename=log_file.format(datetime=start_datetime.replace(':','-')))
    logging.getLogger().setLevel(log_level)
    
    logging.info(f'Parsed args: {json.dumps(dict(args.__dict__), indent=2)}')
    
    if len(args.net.split("_")) == 2 and args.net.split("_")[1] in ["lora","qlora"]:
        print("loading {}...".format(args.net))
        net.load_state_dict(torch.load(args.weights), strict=False)
    else:
        net.load_state_dict(torch.load(args.weights))
    
    net = network_to_half(net)
    print_trainable_parameters(net)
    
    logging.info(net)
    logging.info("\n")
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    
    timings=np.zeros((len(cifar100_test_loader), 1))
    total_time = 0
    
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(tqdm(cifar100_test_loader)):
#             print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
#                 print('GPU INFO.....')
#                 print(torch.cuda.memory_summary(), end='')

            starter.record()
            
#             with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#                 with record_function("model_inference"):
            output = net(image)
            
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            timings[n_iter] = curr_time
            total_time += curr_time
        
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()
    
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    throughput = (n_iter * args.b) / total_time
    logging.info('Average throughput: {}'.format(throughput))
    
    mean_syn = np.sum(timings) / (n_iter+1)
    std_syn = np.std(timings)
    logging.info("Average inference time: {}".format(mean_syn))
    
    if args.gpu:
        logging.info('GPU INFO.....\n')
        logging.info("\n"+torch.cuda.memory_summary())
        logging.info("\n")

    logging.info("Top 1 err: {}\n".format(1 - correct_1 / len(cifar100_test_loader.dataset)))
    logging.info("Top 5 err: {}\n".format(1 - correct_5 / len(cifar100_test_loader.dataset)))
    logging.info("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
