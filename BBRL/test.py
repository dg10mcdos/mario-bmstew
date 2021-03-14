import os

os.environ['OMP_NUM_THREADS'] = '1'
# import argparse
# import torch
# from src.env import create_train_env
# from src.model import PPO
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
# import torch.nn.functional as F
import argparse
import torch
from src.env import MultipleEnvironments, create_train_env
from src.model import PPO
import torch
from src.process import evaluate
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil, csv, time
from src.helpers import flag_get
from src import fullyconnected
from behaviours.feature_extraction import FeatureExtraction
from behaviours.behaviour import Behaviour, VisualMotion
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable

TEST_ON_THE_GO = True


# RIGHT_ONLY = [
#     ['noop'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
# ]
#
#
# # actions for very simple movement
# SIMPLE_MOVEMENT = [
#     ['noop'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
#     ['A'],
#     ['left'],
# ]
#
#
# # actions for more complex movement
# COMPLEX_MOVEMENT = [
#     ['noop'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
#     ['A'],
#     ['left'],
#     ['left', 'A'],
#     ['left', 'B'],
#     ['left', 'A', 'B'],
#     ['down'],
#     ['up'],
# ]

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


# def test(opt):
#     device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
#
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(123)
#     else:
#         torch.manual_seed(123)
#
#     opt.saved_path = os.getcwd() + '/BBRL/tested_model' + opt.saved_path
#     if not os.path.isdir(opt.saved_path):
#         os.makedirs(opt.saved_path)
#     savefile = opt.saved_path + '/PPO_train.csv'
#     print(savefile)
#     title = ['Loops', 'Steps', 'Time', 'AvgLoss', 'MeanReward', "StdReward", "TotalReward", "Flags"]
#     with open(savefile, 'w', newline='') as sfile:
#         writer = csv.writer(sfile)
#         writer.writerow(title)
#     model = fullyconnected.fullyconnected().to(device)
#     model.load_state_dict(torch.load('/home/gerardo/Documents/repos/mario-bm/models/testmodel_upperlayer3f.pth'))
#     envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)
#
#     if torch.cuda.is_available():
#         model.cuda()
#     model.share_memory() # pytorch what it says on the tin.
#     optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
#
#     if TEST_ON_THE_GO:
#         # evaluate(opt, model, envs.num_states, envs.num_actions)
#         mp = _mp.get_context("spawn")
#         process = mp.Process(target=evaluate, args=(opt, model, envs.num_states, envs.num_actions)) # 4 states, 7 actions (buttons)
#         process.start()   ############### broke ###############################
#     curr_states = []  # current state of each env
#     [curr_states.append(env.reset()) for env in envs.envs]
#     curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
#     if torch.cuda.is_available():
#         curr_states = curr_states.cuda()  # chuck it on gpu
#     tot_loops = 0
#     tot_steps = 0
#     while True:
#         start_time = time.time()
#
#         # Accumulate evidence
#         tot_loops += 1
#         old_log_policies = []
#         actions = []  # record of actions
#         values = []  # record of q values
#         states = []  # record of states of processes
#         rewards = []
#         dones = []
#         flags = []
#         for _ in range(opt.num_local_steps):
#             # states.append(curr_states)
#             pass

def test(opt):
    ############ setup pytorch
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(123)
    # else:
    #     torch.manual_seed(123)

    ############ setup save dirs
    opt.saved_path = os.getcwd() + '/BBRL/tested_model' + opt.saved_path
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    savefile = opt.saved_path + '/PPO_train.csv'
    print(savefile)
    title = ['Loops', 'Steps', 'Time', 'AvgLoss', 'MeanReward', "StdReward", "TotalReward", "Flags"]
    with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

    ############ Create environment

    RIGHT_ONLY = [
        ['noop'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'], ]
    SIMPLE_MOVEMENT = [
        ['noop'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'], ]
    COMPLEX_MOVEMENT = [
        ['noop'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'],
        ['left', 'A'],
        ['left', 'B'],
        ['left', 'A', 'B'],
        ['down'],
        ['up'], ]
    env = create_train_env(COMPLEX_MOVEMENT, mp_wrapper=False, )
    env.reset()

    ############ load models & optimiser

    print("loading upper layer net...\n")

    model = fullyconnected.fullyconnected().to(device)
    model.load_state_dict(torch.load('/home/gerardo/Documents/repos/mario-bm/models/testmodel_upperlayer3f.pth'))

    print("loaded.\nloading feature extraction...\n")
    feat_extract = FeatureExtraction(device)
    feat_extract.load_state_dict(torch.load("/home/gerardo/Documents/repos/mario-bm/models/bestmodel_ae_128x16x16.pth"))
    feat_extract.to(device)

    print("loaded.\nloading behaviours & motion...\n")
    button_list = ['a', 'b', 'l', 'r', 'u', 'd']
    motion_list = [None] * 6
    b_list = [None] * 6
    for i in range(0, len(button_list)):
        motion_list[i] = VisualMotion(3).to(device)
        motion_list[i].load_state_dict(torch.load(
            '/home/gerardo/Documents/repos/mario-bm/models/bestmodel_vmt_' + "bx" + str(button_list[i]) + str(3) +
            'f.pth'))
        for param in motion_list[i].parameters():
            param.requires_grad = False
        motion_list[i].eval()

        b_list[i] = Behaviour().to(device)
        b_list[i].load_state_dict(torch.load(
            '/home/gerardo/Documents/repos/mario-bm/models/bestmodel_bx_' + "bx" + str(button_list[i]) + str(3) +
            'f.pth'))
        for param in b_list[i].parameters():
            param.requires_grad = False
        b_list[i].eval()

    for param in feat_extract.parameters():
        param.requires_grad = False
    feat_extract.eval()

    for param in model.parameters():  # <-- freeze upper layer
        param.requires_grad = False
    model.eval()
    params = list(model.parameters())
    ############ test
    # state, reward, done, info = env.step(x) x = action in movement array
    state, reward, done, info, new_states = env.step(0)  # state should be 3x[w:240,h:224,rgb:3] 1,672,240,3
    transform2apply = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    env.render()
    while True:  # run right
        time.sleep(0.04)
        state = state.squeeze(0)
        state = state.reshape((3, 224, 240, 3))
        imgarrays = []
        for i in range(0, len(new_states)):
            img = Image.fromarray(new_states[i]).convert("RGB")
            imgarrays.append(transform2apply(img).unsqueeze(0).to(device))
        feats = []
        for img in imgarrays:
            feats.append(feat_extract.encode(img))
        features = torch.cat(tuple(feats), dim=1)
        motions = []
        for i in range(0, len(motion_list)):
            motions.append(motion_list[i](features))
        prob_buttons = model(motions[0]).squeeze(0)
        b_result = []
        for i in range(0, len(button_list)):
            b_result.append(b_list[i](motions[i]))
        b_to_b_dict = {
            'a': 'A',
            'b': 'B',
            'l': 'left',
            'r': 'right',
            'u': 'up',
            'd': 'down'
        }
        push = []
        for i in range(0, len(button_list)):
            if prob_buttons[i] > 0.1 and b_result[i] > 0.1:
                push.append(b_to_b_dict[button_list[i]])
        action = 0
        for i in range(0,len(COMPLEX_MOVEMENT)):
            if set(COMPLEX_MOVEMENT[i]) == set(push):
                # print("Match")
                # print(f"push: {push}, COMPLEX_MOVEMENT: {COMPLEX_MOVEMENT[i]}, index: {i}")
                action = i

        state, reward, done, info, new_states = env.step(action)
        env.render()
        if done == True:
            env.reset()
            state, reward, done, info, new_states = env.step(0)  # state should be 3x[w:240,h:224,rgb:3] 1,672,240,3


# def test(opt):
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(123)
#     else:
#         torch.manual_seed(123)
#     if opt.action_type == "right":
#         actions = RIGHT_ONLY
#     elif opt.action_type == "simple":
#         actions = SIMPLE_MOVEMENT
#     else:
#         actions = COMPLEX_MOVEMENT
#     env = create_train_env(opt.world, opt.stage, actions,
#                            "{}/video_{}_{}.mp4".format(opt.output_path, opt.world, opt.stage))
#     model = PPO(env.observation_space.shape[0], len(actions))
#     if torch.cuda.is_available():
#         model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage)))
#         model.cuda()
#     else:
#         model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage),
#                                          map_location=lambda storage, loc: storage))
#     model.eval()
#     state = torch.from_numpy(env.reset())
#     while True:
#         if torch.cuda.is_available():
#             state = state.cuda()
#         logits, value = model(state)
#         policy = F.softmax(logits, dim=1)
#         action = torch.argmax(policy).item()
#         state, reward, done, info = env.step(action)
#         state = torch.from_numpy(state)
#         env.render()
#         if info["flag_get"]:
#             print("World {} stage {} completed".format(opt.world, opt.stage))
#             break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
