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
from BBRL import train
'''
torch.save(net_motion.state_dict(), './models/bestmodel_vmt_' + namebx + str(net_motion.noframes) + 'f.pth')
            torch.save(net_bx.state_dict(), './models/bestmodel_bx_' + namebx + str(net_motion.noframes) + 'f.pth')
'''
matplotlib.use('Agg')


def load_experiment(filename):
    with open(filename) as json_file:
        opt = json.load(json_file)
                                                                # current path: /home/gerardo/Documents/repos/mario-bm
    opt['feat_path'] = os.path.abspath(opt['feat_path'])        # "./models/bestmodel_ae_128x16x16.pth"
    opt['data_path'] = os.path.abspath(opt['data_path'])        # "./data/"
    opt['bx_data_path'] = os.path.abspath(opt['bx_data_path'])  # "./bx_data/gerardo120719_3f.csv"
    return opt

# ['a', 'b', 'l', 'r', 'u', 'd']
if __name__ == "__main__":
    # Generate dataset
    opt = load_experiment("./" + sys.argv[1])  # pass in json experiment file from command line

    if os.path.isfile(opt["bx_data_path"]):  # if generated then use
        print("Using existing file: " + opt["bx_data_path"])
    else:  # else call from gencsv_behaviour.py
        gendb = GenerateDatasetBx(opt["dataset"], opt["data_path"], opt["no_frames"], opt["bx_data_path"])
        gendb.run()

    # Setting up PyTorch stuff
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    print("CUDA Available: ", torch.cuda.is_available())

    device = torch.device(
        "cuda" if (opt['use_cuda'] and torch.cuda.is_available()) else "cpu")  # gpu/cpu - torch object

    # Feature extraction - assumes you have a trained fe network (feature_extraction.py)
    feat_extract = FeatureExtraction(device)
    feat_extract.load_state_dict(torch.load(opt['feat_path']))
    feat_extract.to(device)

    transform2apply = transforms.Compose([transforms.Resize((opt['img_size'], opt['img_size'])), transforms.ToTensor()])
    print("loading behaviours...\n")
    motion_list = [None] * 6
    b_list = [None] * 6
    for i in range(0, len(opt["buttons"])):
        motion_list[i] = VisualMotion(opt["no_frames"]).to(device)
        motion_list[i].load_state_dict(torch.load('./models/bestmodel_vmt_' + "bx" + str(opt["buttons"][i]) + str(opt['no_frames']) + 'f.pth'))
        b_list[i] = Behaviour().to(device)
        b_list[i] = b_list[i].load_state_dict(torch.load('./models/bestmodel_bx_' + "bx" + str(opt["buttons"][i]) + str(opt["no_frames"]) + 'f.pth'))
    print("loaded behaviours successfully!\n")
    print("training...\n")
    train.train(opt, b_list,motion_list)


