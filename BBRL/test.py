import os
from torchsummary import summary

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
from src.PPOLike import PPOLike

TEST_ON_THE_GO = True


def check_flag(info):
    out = 0
    for i in info:
        if flag_get(i):
            out += 1
    return out


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
    parser.add_argument("--stage", type=int, default=2)
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


def test(opt):
    ############ setup pytorch
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    r_taken = 0
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
        ['up'], ['B'], ['A', 'B']]
    env = create_train_env(COMPLEX_MOVEMENT, mp_wrapper=False, )
    env.reset()

    ############ load models & optimiser

    print("loading upper layer net...\n")

    model = fullyconnected.fullyconnected().to(device)
    # model = PPOLike.PPOLike().to(device)
    model.load_state_dict(torch.load('/home/gerardo/Documents/repos/mario-bm/models/testmodel_upperlayer3f.pth'))

    print("loaded.\nloading feature extraction...\n")
    feat_extract = FeatureExtraction(device)
    feat_extract.load_state_dict(torch.load("/home/gerardo/Documents/repos/mario-bm/models/bestmodel_ae_128x16x16.pth"))
    feat_extract.to(device)
    print(feat_extract)
    print("SUMMARY: ", summary(feat_extract, input_size=(3,256,256)))

    print("loaded.\nloading behaviours & motion...\n")
    # button_list = ['a', 'b', 'l',  'r', 'u', 'd']
    button_list = ['a', 'b', 'l',  'r', 'u', 'd']

    motion_list = [None] * len(button_list)
    b_list = [None] * len(button_list)
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
    # print(state.shape)
    reward_list = []

    env.render()
    # for i in range(0,8):
    #     state, reward, done, info, new_states = env.step(1)  # state should be 3x[w:240,h:224,rgb:3] 1,672,240,3
    #     reward_list.append(reward)
    #
    #     env.render()

    time_for_random = 0
    reward_list.append(reward)
    while True:  # run right
        # time.sleep(0.007)
        state = state.squeeze(0)
        state = state.reshape((3, 224, 240, 3))
        imgarrays = []
        for i in range(0, len(new_states)):
            img = Image.fromarray(new_states[i]).convert("RGB")
            imgarrays.append(transform2apply(img).unsqueeze(0).to(device))
        feats = []
        for img in imgarrays:
            feats.append(feat_extract.encode(img))
        # print(feats[0].shape)
        features = torch.cat(tuple(feats), dim=1)
        motions = []
        for i in range(0, len(motion_list)):
            motions.append(motion_list[i](features))
        prob_buttons = model(motions[0]).squeeze(0)
        b_result = []
        for i in range(0, len(button_list)):
            b_result.append(b_list[i](motions[i]))

        # button_list = ['A', 'B', 'left', 'right', 'up', 'down']
        button_list = ['A', 'B',  'left', 'right', 'up', 'down']

        push = []
        for i in range(0, len(button_list)):
            # print(button_list[i], prob_buttons[i], b_result[i])
            if prob_buttons[i] > 0.04 and b_result[i] > 0.12: # 0.04 0.12

                push.append(button_list[i])

        random_action = 0
        # print(np.sum(reward_list[-20:]))
        if(np.sum(reward_list[-20:]) < 1  and r_taken==0):
            random_action = np.random.randint(0,len(button_list))
            if(button_list[random_action] not in set(push)):
                push.append(button_list[random_action])
                r_taken = 15
            else:
                random_action = 0
        if r_taken > 0 and reward==0:
            push.append(button_list[random_action])
            r_taken-=1


        action = 0
        for i in range(0, len(COMPLEX_MOVEMENT)):
            if set(COMPLEX_MOVEMENT[i]) == set(push):
                # print("Match")
                # print(f"push: {push}, COMPLEX_MOVEMENT: {COMPLEX_MOVEMENT[i]}, index: {i}")
                action = i
            else:
                pass
        if action == 0:
            # print("no match")
            action = 3
            pass
        state, reward, done, info, new_states = env.step(action)
        # time.sleep(0.05)
        time.sleep(0.2)

        reward_list.append(reward)

        env.render()

        if done == True:
            print("Ended in position ", info['x_pos']*256+info['x_pos_2'])
            time.sleep(3)
            env.reset()
            state, reward, done, info, new_states = env.step(0)  # state should be 3x[w:240,h:224,rgb:3] 1,672,240,3


def PPOmodetrain(opt):
    ############ setup pytorch
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

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
        ['up'],
        ['A', 'B']]
    env = create_train_env(COMPLEX_MOVEMENT, mp_wrapper=False, )
    env.reset()

    ############ load models & optimiser

    print("loading upper layer net...\n")

    model = PPOLike().to(device)
    # model.load_state_dict(torch.load('/home/gerardo/Documents/repos/mario-bm/models/testmodel_upperlayer3f.pth'))

    print("loaded.\nloading feature extraction...\n")
    feat_extract = FeatureExtraction(device)
    feat_extract.load_state_dict(torch.load("/home/gerardo/Documents/repos/mario-bm/models/bestmodel_ae_128x16x16.pth"))
    feat_extract.to(device)

    print("loaded.\nloading behaviours & motion...\n")
    button_list = ['a', 'b', 'l', 'r', 'u', 'd']
    motion_list = [None] * len(button_list)
    b_list = [None] * len(button_list)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    tot_loops = 0
    tot_steps = 0

    # for param in model.parameters():  # <-- freeze upper layer
    #     param.requires_grad = False
    # model.eval()
    params = list(model.parameters())
    ############ test
    # state, reward, done, info = env.step(x) x = action in movement array
    state, reward, done, info, new_states = env.step(0)  # state should be 3x[w:240,h:224,rgb:3] 1,672,240,3
    env.render()
    transform2apply = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
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
    b_result = []
    for i in range(0, len(button_list)):
        b_result.append(b_list[i](motions[i]))
    # curr_states = torch.from_numpy(np.concatenate(motions[0], 0))
    state = motions[0]
    curr_states = state
    button_list = ['A', 'B', 'left', 'right', 'up', 'down']

    while True:  # run right

        start_time = time.time()

        tot_loops += 1
        old_log_policies = []
        actions = []  # record of actions
        values = []  # record of q values
        states = []  # record of states of processes
        rewards = []
        dones = []
        flags = []
        for _ in range(opt.num_local_steps):

            states.append(curr_states)
            logits, value = model(curr_states)  # actor, critic
            print("logits: ", logits,"shape: ",logits.shape)
            values.append(value.squeeze())  # critic [4,1] -> 4

            policy = F.softmax(logits, dim=1)  # turns action scores into probabilities
            print("Policy: ", policy, "shape", policy.shape)
            # ['noop'],
            # ['right'],
            # ['right', 'A'],
            # ['right', 'B'],
            # ['right', 'A', 'B'],
            # ['A'],
            # ['left'],
            # ['left', 'A'],
            # ['left', 'B'],
            # ['left', 'A', 'B'],
            # ['down'],
            # ['up'],]
            old_m = Categorical(policy)  # buttons with probabilities
            print(old_m)
            b_to_b_dict = {
                'a': 'A',
                'b': 'B',
                'l': 'left',
                'r': 'right',
                'u': 'up',
                'd': 'down'
            }
            push = []
            probabilities_squeezed = policy.squeeze(0)
            for i in range(0, len(button_list)):
                print("prob: ", probabilities_squeezed[i].item(), "b_result: ", b_result[i], "button: ", button_list[i])
                if probabilities_squeezed[i].item() > 0.1 and b_result[i].item() > 0.1:
                    push.append(button_list[i])
            action = 0
            for i in range(0, len(COMPLEX_MOVEMENT)):
                print(push)
                if set(COMPLEX_MOVEMENT[i]) == set(push):
                    print("Match")
                    print(f"push: {push}, COMPLEX_MOVEMENT: {COMPLEX_MOVEMENT[i]}, index: {i}")
                    action = i
            actions.append(action)  # record action
            print(torch.tensor(action))
            old_log_policy = old_m.log_prob(action)  # probability of action
            old_log_policies.append(old_log_policy)  # record action probability

            state, reward, done, info, new_states = env.step(action)

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

            # state = torch.from_numpy(np.concatenate(motions[0], 0))
            state = motions[0]
            curr_states = state

            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            rewards.append(reward)
            dones.append(done)
            # flags.append(check_flag(info) / opt.num_processes)

        # Training stage

        _, next_value = model(motions[0])  # actor, critic
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()  # detach?
        states = torch.cat(states)
        gae = 0  # generalised advantage estimator?
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)

        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        avg_loss = []
        for _ in range(opt.num_epochs):
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            for j in range(opt.batch_size):
                batch_indices = indice[
                                int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                                        opt.num_local_steps * opt.num_processes / opt.batch_size))]
                logits, value = model(states[batch_indices])  ########################
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                   torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                   advantages[batch_indices]))
                # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                optimizer.zero_grad()
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                avg_loss.append(total_loss.cpu().detach().numpy().tolist())

        avg_loss = np.mean(avg_loss)
        all_rewards = torch.cat(rewards).cpu().numpy()
        tot_steps += opt.num_local_steps * opt.num_processes
        sum_reward = np.sum(all_rewards)
        mu_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        any_flags = np.sum(flags)
        ep_time = time.time() - start_time
        # data = [tot_loops, tot_steps, ep_time, avg_loss, mu_reward, std_reward, sum_reward, any_flags]
        data = [tot_loops, tot_steps, "{:.6f}".format(ep_time), "{:.4f}".format(avg_loss), "{:.4f}".format(mu_reward),
                "{:.4f}".format(std_reward), "{:.2f}".format(sum_reward), any_flags]

        with open(savefile, 'a', newline='') as sfile:
            writer = csv.writer(sfile)
            writer.writerows([data])
        print("Steps: {}. Total loss: {} Loops: {}".format(tot_steps, total_loss, tot_loops))
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
    # PPOmodetrain(opt)