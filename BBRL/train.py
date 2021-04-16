import os
import pickle

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from src.env import MultipleEnvironments, create_train_env
from src.model import PPO
from src.process import evaluate
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil, csv, time
from src.helpers import flag_get
from behaviours.feature_extraction import FeatureExtraction
from behaviours.behaviour import Behaviour, VisualMotion, trainNetBx
from behaviours.mariodataloader import DatasetMarioFc

from behaviours.utils import create_datasets_split
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from behaviours.behaviour import VisualMotion
from src.fullyconnected import fullyconnected
from behaviours import plot
TEST_ON_THE_GO = True
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Number of concurrent processes, has to be larger than 1")
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


'''
how they were saved in behaviour.py:

torch.save(net_motion.state_dict(), './models/bestmodel_vmt_' + namebx + str(net_motion.noframes) + 'f.pth')
            torch.save(net_bx.state_dict(), './models/bestmodel_bx_' + namebx + str(net_motion.noframes) + 'f.pth')
how to load model:            
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
'''

def check_flag(info):
    out = 0
    for i in info:
        if flag_get(i):
            out += 1
    return out


def flatten_list(losses_list):
    flat_list = [item for sublist in losses_list for item in sublist]
    return flat_list


def save_stats(stats_train, stats_val):
    save_dir = os.path.abspath('./plots/blayer')
    with open(save_dir + "/" + 'stats_train.pkl', 'wb') as fp:
        pickle.dump(stats_train, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_dir + "/" + 'stats_val.pkl', 'wb') as fp:
        pickle.dump(stats_val, fp, protocol=pickle.HIGHEST_PROTOCOL)


def train_fully_connected(opt, b_list, motion_list, feat_extract, train_loader, val_loader, n_epochs, lr, device,
                          fcnet):  # opt is object storing args
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    opt.saved_path = os.getcwd() + '/BBRL/' + opt.saved_path
    # freeze the behaviour, motion and feat_extract nets
    for param in feat_extract.parameters():
        param.requires_grad = False
    feat_extract.eval()
    for net in b_list:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()
    for net in motion_list:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()
    n_batches = len(train_loader)
    params = list(fcnet.parameters())
    optimizer_fc = optim.Adam(params, lr=lr)
    scheduler = StepLR(optimizer_fc, step_size=5, gamma=0.8)
    lr_step = optimizer_fc.param_groups[0]['lr']
    training_start_time = time.time()

    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", lr)
    print("=" * 30)

    prev_loss = np.inf
    stats_train = []
    stats_val = []

    for epoch in range(n_epochs):
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        losses_list = []
        for i, data in enumerate(train_loader, 0):
            inputs = data['image']
            controller = data['state'].to(device)
            controller = Variable(controller)
            feats = []
            for j in range(0, len(inputs)):  # in this example 3
                image = Variable(inputs[j].to(device))
                feats.append(feat_extract.encode(image))
            features = torch.cat(tuple(feats), dim=1)
            optimizer_fc.zero_grad()

            motions = []
            for j in range(0, len(motion_list)):
                motions.append(motion_list[j](features))
            behaviours = fcnet(motions[0])# using only first motion as example. hopefully should be same
            loss = fcnet.loss_function(behaviours, controller)
            loss.backward()
            optimizer_fc.step()
            mse = nn.MSELoss(reduction='none')(behaviours, controller)
            losses_list.append(mse.data.cpu().numpy().tolist())
            total_train_loss += loss.item()
            if (i + 1) % (print_every + 1) == 0:  # print 9 for each epoch
                flat_list = flatten_list(losses_list)
                flat_list = np.array(flat_list)  # 18 * 256
                mu = np.mean(flat_list)
                standev = np.std(flat_list)
                var = np.var(flat_list)
                stats_train.append([epoch, mu, standev, var])

                print(
                    '{:d}% - Epoch [{}/{}], Step [{}/{}], MSE mean: {:.4f}, std: {:.4f}, var: {:.4f}. Took: {:.2f}s with LR: {:.5f}'
                        .format(int(100 * (i + 1) / n_batches), epoch + 1, n_epochs, i + 1, len(train_loader), mu,
                                standev,
                                var, time.time() - start_time, lr_step))
                if epoch > 0:
                    save_dir = './plots/blayer'
                    plot.plot(save_dir, '/loss train', mu)
                    plot.flush()
                    plot.tick()
                # Reset running loss and time
                start_time = time.time()
                losses_list = []
        total_train_loss = total_train_loss / len(train_loader)
        total_val_loss, valmu, valstdev, valvar = validation_and_plots(feat_extract, b_list[0], motion_list[0],
                                                                       val_loader, device, fcnet)  # validate ur model
        stats_val.append([epoch, valmu, valstdev, valvar])
        if prev_loss > total_val_loss:  # save models (neural network) if better than previous
            torch.save(fcnet.state_dict(), './models/testmodel_upperlayer' + str(motion_list[0].noframes) + "f.pth")
            prev_loss = total_val_loss

        scheduler.step()
        lr_step = optimizer_fc.param_groups[0]['lr']
        save_stats(stats_train, stats_val)  ############################################################
    print("Training finished, took {:.2f} hr".format((time.time() - training_start_time) / 3600))
    return fcnet




def validation_and_plots(feat_extract, net_bx, net_motion, val_loader, device, fcnet, prefix='./plots'):
    print('===========================================')
    n_batches = len(val_loader)
    print_every = n_batches // 10

    total_val_loss = 0
    losses_list = []
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs = data['image']
            target = data['state'].to(device)
            target = Variable(target)
            feats = []
            for j in range(0, len(inputs)):  # in this example 3
                image = Variable(inputs[j].to(device))
                feats.append(feat_extract.encode(image))
            features = torch.cat(tuple(feats), dim=1)
            net_bx.eval()
            net_motion.eval()
            fcnet.eval()
            output_motion = net_motion(features)
            output = fcnet(output_motion)
            val_loss_size = fcnet.loss_function(output, target)
            total_val_loss += val_loss_size.data.item()
            mse = nn.MSELoss(reduction='none')(output, target)
            losses_list.append(mse.data.cpu().numpy().tolist())

            if (i + 1) % (print_every + 1) == 0:
                print('Validation {:d}% - Step [{}/{}]'.format(int(100 * (i + 1) / n_batches), i + 1, len(val_loader)))
    flat_list = flatten_list(losses_list)
    flat_list = np.array(flat_list)
    mu = np.mean(flat_list)
    standev = np.std(flat_list)
    var = np.var(flat_list)
    print(
        "Validation loss mean: {:.4f} (or {:.4f}), std {:.4f}, var {:.4f}".format(total_val_loss / len(val_loader), mu,
                                                                                  standev, var))
    print("Validation completed!!!")
    print('===========================================')
    return total_val_loss / len(val_loader), mu, standev, var


if __name__ == "__main__":
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    button_list = ['a', 'b', 'l', 'r', 'u', 'd']
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    print("loading feature extraction...\n")
    feat_extract = FeatureExtraction(device)
    feat_extract.load_state_dict(torch.load("/home/gerardo/Documents/repos/mario-bm/models/bestmodel_ae_128x16x16.pth"))
    feat_extract.to(device)
    print("loaded.\nloading behaviours & motion...\n")
    # print(feat_extract)
    motion_list = [None] * 6
    b_list = [None] * 6
    for i in range(0, len(button_list)):
        motion_list[i] = VisualMotion(3).to(device)
        motion_list[i].load_state_dict(
            torch.load(
                '/home/gerardo/Documents/repos/mario-bm/models/bestmodel_vmt_' + "bx" + str(button_list[i]) + str(
                    3) + 'f.pth'))
        b_list[i] = Behaviour().to(device)
        b_list[i].load_state_dict(
            torch.load('/home/gerardo/Documents/repos/mario-bm/models/bestmodel_bx_' + "bx" + str(button_list[i]) + str(
                3) + 'f.pth'))
    print("loaded.\nretrieving arguments...\n")
    opt = get_args()
    print("loaded.\nretrieved. loading data...\n")

    transform2apply = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = DatasetMarioFc(file_path="/home/gerardo/Documents/repos/mario-bm",
                             csv_name="/home/gerardo/Documents/repos/mario-bm/bx_data/allitems.csv",
                             transform_in=transform2apply)
    train_loader, validation_loader = create_datasets_split(dataset, True, 0.8, 256, 128)
    fcnet = fullyconnected().to(device)

    print("loaded.\ntraining...")
    fcnet = train_fully_connected(opt, b_list, motion_list, feat_extract, train_loader, validation_loader, 60,
                                  0.001, device,
                                  fcnet)
    # for i in range(3,10):
    #     fcnet = train_fully_connected(opt, b_list, motion_list, feat_extract, train_loader, validation_loader, i*10, 0.001, device,
    #                           fcnet)
    #     torch.save(fcnet.state_dict(), './models/trial/epoc' + i)

# if __name__ == "__main__":
#     opt = get_args()
#     print(opt)
#     train(opt)
