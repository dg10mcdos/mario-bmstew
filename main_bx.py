# Main file compiles a dataset and trains behaviours, see point 4 in README. Make sure to change the name of the experiment file in line 35
# for the file of the current experiment. In the JSON experiment file, you can define how many frames the dataset generation should skip. This trick helps
# the network to not overfit the data (need futher experimentation though)
# Author: Gerardo Aragon-Camarasa, 2020
# Training behaviours e.g. pressing each button

'''
must use experiment file
'''

import json
import os
import sys

import matplotlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from behaviours.behaviour import Behaviour, VisualMotion, trainNetBx
from behaviours.feature_extraction import FeatureExtraction
from behaviours.gencsv_behaviour import GenerateDatasetBx
from behaviours.mariodataloader import DatasetMarioBxv2
from behaviours.utils import create_datasets_split

matplotlib.use('Agg')


def load_experiment(filename):
    with open(filename) as json_file:
        opt = json.load(json_file)
                                                                # current path: /home/gerardo/Documents/repos/mario-bm  
    opt['feat_path'] = os.path.abspath(opt['feat_path'])        # "./models/bestmodel_ae_128x16x16.pth"
    opt['data_path'] = os.path.abspath(opt['data_path'])        # "./data/"
    opt['bx_data_path'] = os.path.abspath(opt['bx_data_path'])  # "./bx_data/gerardo120719_3f.csv"
    return opt


if __name__ == "__main__":
                                                                # Generate dataset
    opt = load_experiment("./"+sys.argv[1])                     # pass in json experiment file from command line
                                                        
    if os.path.isfile(opt["bx_data_path"]):                     # if generated then use
        print("Using existing file: "+opt["bx_data_path"])
    else:                                                       # else call from gencsv_behaviour.py 
        gendb = GenerateDatasetBx(opt["dataset"], opt["data_path"], opt["no_frames"], opt["bx_data_path"])
        gendb.run()

    # Setting up PyTorch stuff
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    print("CUDA Available: ",torch.cuda.is_available())

    device = torch.device("cuda" if (opt['use_cuda'] and torch.cuda.is_available()) else "cpu")     # gpu/cpu - torch object
  


    # Feature extraction - assumes you have a trained fe network (feature_extraction.py) 
    feat_extract = FeatureExtraction(device)                    
    feat_extract.load_state_dict(torch.load(opt['feat_path']))  
    feat_extract.to(device)

    transform2apply = transforms.Compose([transforms.Resize((opt['img_size'],opt['img_size'])), transforms.ToTensor()])

    # ********************************
    # Train Behaviours!
    print(opt)
    for eachBx in opt["buttons"]:

        print("Training " + eachBx + "...\n")
        dataset = DatasetMarioBxv2(csv_name=opt["bx_data_path"], buttontrain=eachBx, transform_in=transform2apply)  # call to mariodataloader, 
        # returns [
        # [frame_sequence_number, images_in_this sequence_converted & transformed for NN, probability of the button being trained being pressed]
        # [frame_sequence_number2, images_in_this sequence_converted & transformed for NN, probability of the button being trained being pressed]
        # ...
        # ]
        # mariodataloader.py goes into ./data/runs/images & states, then loads this data in a way
        # that can be parsed into a neural network

        train_loader, validation_loader = create_datasets_split(dataset, opt['shuffle_dataset'], opt['validation_split'], opt['batch_train'], opt['batch_validation'])  # call to helper function in utils.py
        net_motion = VisualMotion(opt["no_frames"]).to(device)
        net_bx = Behaviour().to(device)
        trainNetBx(net_motion, net_bx, feat_extract, train_loader, validation_loader, opt['epochs'], opt['learning_rate'], device, "bx"+eachBx)  # call to behaviour.py
        print("Training "+eachBx+" done!")
    # ********************************

    print("Done!")
