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


# def train(opt): # opt is object storing args
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(123)
#     else:
#         torch.manual_seed(123)
#
#     opt.saved_path = os.getcwd() + '/baselines/PPO/' + opt.saved_path
#
#
#     if not os.path.isdir(opt.saved_path):
#         os.makedirs(opt.saved_path)
#
#     savefile = opt.saved_path + '/PPO_train.csv'
#     print(savefile)
#     title = ['Loops', 'Steps', 'Time', 'AvgLoss', 'MeanReward', "StdReward", "TotalReward", "Flags"]
#     with open(savefile, 'w', newline='') as sfile:
#         writer = csv.writer(sfile)
#         writer.writerow(title)
#
#     # Create environments
#     envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)
#     # Create model and optimizer
#     model = PPO(envs.num_states, envs.num_actions) # 4 states(assuming processes), 7 actions (buttons)
#     if torch.cuda.is_available():
#         model.cuda()
#     model.share_memory() # pytorch what it says on the tin.
#     optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
#
#     if TEST_ON_THE_GO:
#         # evaluate(opt, model, envs.num_states, envs.num_actions)
#         mp = _mp.get_context("spawn")
#         process = mp.Process(target=evaluate, args=(opt, model, envs.num_states, envs.num_actions)) # 4 states, 7 actions (buttons)
#         process.start()
#     # Reset envs
#     curr_states = [] # current state of each env
#     [curr_states.append(env.reset()) for env in envs.envs]
#
#     curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
#     if torch.cuda.is_available():
#         curr_states = curr_states.cuda() # chuck it on gpu
#
#     tot_loops = 0
#     tot_steps = 0
#
#     # Start main loop (EXECUTED AFTER EACH ACTION)
#     while True:
#         # Save model each loop
#         # if tot_loops % opt.save_interval == 0 and tot_loops > 0:
#         #     # torch.save(model.state_dict(), "{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
#         #     torch.save(model.state_dict(), "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, tot_loops))
#
#         start_time = time.time()
#
#         # Accumulate evidence
#         tot_loops += 1
#         old_log_policies = []
#         actions = [] # record of actions
#         values = [] # record of q values
#         states = [] # record of states of processes
#         rewards = []
#         dones = []
#         flags = []
#         for _ in range(opt.num_local_steps):
#             # From given states, predict an action
#             states.append(curr_states)
#
#             logits, value = model(curr_states) # actor, critic
#
#             # actor [4,7] "score" for pressing buttons aka given our state probability of each action
#             # critic [4,1] q-value for state
#             values.append(value.squeeze()) # critic [4,1] -> 4
#             policy = F.softmax(logits, dim=1) # turns action scores into probabilities
#             old_m = Categorical(policy) # actions with probabilities
#             action = old_m.sample() # choose random action wrt probabilities
#
#
#
#             actions.append(action) # record action
#             old_log_policy = old_m.log_prob(action) # probability of action
#             old_log_policies.append(old_log_policy) # record action probability
#             # Evaluate predicted action
#             result = []
#             # print(action.shape)
#             # print(action[0].cpu().item())
#             # ac = action.cpu().item()
#             if torch.cuda.is_available():
#                 # [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
#                 [result.append(env.step(act.item())) for env, act in zip(envs.envs, action.cpu())] # take action,
#             else:
#                 #[agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]
#                 [result.append(env.step(act.item())) for env, act in zip(envs.envs, action)]
#                 # result :
#                 # state[array(28224)],
#                 # reward - number
#                 # done - boolean
#                 # info - [{'levelLo': 0, 'enemyType5': 0, 'xscrollLo': 16, 'floatState': 0, 'enemyType4': 0, 'status': 8, 'levelHi': 0, 'score': 0, 'lives': 2,
#                 # 'scrollamaount': 1, 'scrolling': 16, 'xscrollHi': 0, 'enemyType2': 0, 'time': 397, 'enemyType3': 0, 'coins': 0, 'gameMode': 1, 'enemyType1': 0}]
#             state, reward, done, info = zip(*result)
#             state = torch.from_numpy(np.concatenate(state, 0))
#
#             if torch.cuda.is_available():
#                 state = state.cuda()
#                 reward = torch.cuda.FloatTensor(reward)
#                 done = torch.cuda.FloatTensor(done)
#             else:
#                 reward = torch.FloatTensor(reward)
#                 done = torch.FloatTensor(done)
#
#             rewards.append(reward)
#             dones.append(done)
#             flags.append(check_flag(info) / opt.num_processes)
#             curr_states = state
#
#         # Training stage
#         _, next_value, = model(curr_states) # retrieve next q value
#         next_value = next_value.squeeze()
#         old_log_policies = torch.cat(old_log_policies).detach()
#         # print(actions[0][:])
#         actions = torch.cat(actions)
#         values = torch.cat(values).detach() # detach?
#         states = torch.cat(states)
#         gae = 0 # generalised advantage estimator?
#         R = []
#         for value, reward, done in list(zip(values, rewards, dones))[::-1]:
#             gae = gae * opt.gamma * opt.tau
#             gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
#             next_value = value
#             R.append(gae + value)
#
#         R = R[::-1]
#         R = torch.cat(R).detach()
#         advantages = R - values
#         avg_loss = []
#         for _ in range(opt.num_epochs):
#             indice = torch.randperm(opt.num_local_steps * opt.num_processes)
#             for j in range(opt.batch_size):
#                 batch_indices = indice[int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
#                                         opt.num_local_steps * opt.num_processes / opt.batch_size))]
#                 logits, value = model(states[batch_indices])
#                 new_policy = F.softmax(logits, dim=1)
#                 new_m = Categorical(new_policy)
#                 new_log_policy = new_m.log_prob(actions[batch_indices])
#                 ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
#                 actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices], torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) * advantages[batch_indices]))
#                 # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
#                 critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
#                 entropy_loss = torch.mean(new_m.entropy())
#                 total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
#                 optimizer.zero_grad()
#                 total_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#                 optimizer.step()
#                 avg_loss.append(total_loss.cpu().detach().numpy().tolist())
#
#         avg_loss = np.mean(avg_loss)
#         all_rewards = torch.cat(rewards).cpu().numpy()
#         tot_steps += opt.num_local_steps * opt.num_processes
#         sum_reward = np.sum(all_rewards)
#         mu_reward = np.mean(all_rewards)
#         std_reward = np.std(all_rewards)
#         any_flags = np.sum(flags)
#         ep_time = time.time() - start_time
#         # data = [tot_loops, tot_steps, ep_time, avg_loss, mu_reward, std_reward, sum_reward, any_flags]
#         data = [tot_loops, tot_steps, "{:.6f}".format(ep_time), "{:.4f}".format(avg_loss), "{:.4f}".format(mu_reward), "{:.4f}".format(std_reward), "{:.2f}".format(sum_reward), any_flags]
#
#         with open(savefile, 'a', newline='') as sfile:
#             writer = csv.writer(sfile)
#             writer.writerows([data])
#         print("Steps: {}. Total loss: {} Loops: {}".format(tot_steps, total_loss, tot_loops))

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
    button_list = ['a', 'b', 'l', 'r', 'u', 'd']

    n_batches = len(train_loader)
    params = list(fcnet.parameters())
    optimizer_fc = optim.Adam(params, lr=lr)
    scheduler = StepLR(optimizer_fc, step_size=5, gamma=0.4)
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

            # behaviours = torch.empty(256, 7, dtype=torch.float32)
            # for i in range(0, len(b_list)):
            #     output = b_list[i](motions[i])
            #     output = torch.flatten(output)
            #     behaviours[:, i] = output
            behaviours = fcnet(motions[0])  # using only first motion as example. hopefully should be same
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

    # SIMPLE_MOVEMENT = [
    #     ['noop'],
    #     ['right'],
    #     ['right', 'A'],
    #     ['right', 'B'],
    #     ['right', 'A', 'B'],
    #     ['A', 'B'],
    #     ['left'],
    # ]
    #
    # env = create_train_env(SIMPLE_MOVEMENT, mp_wrapper=False)
    #
    # env.reset()
    # state, reward, done, info = env.step(1)
    # while True:
    #     # i, y, u, j = env.step(np.random.randint(0,7))
    #     i, y, u, j = env.step(5)
    #     i, y, u, j = env.step(5)
    #
    #     i, y, u, j = env.step(1)
    #     env.render()
    #     print(y)


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
    print("loaded.\n loading behaviours & motion...\n")
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
    print("loaded. retrieving arguments...\n")
    opt = get_args()
    print("retrieved. loading data...\n")

    transform2apply = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = DatasetMarioFc(file_path="/home/gerardo/Documents/repos/mario-bm",
                             csv_name="/home/gerardo/Documents/repos/mario-bm/bx_data/gerardo120719_3f.csv",
                             transform_in=transform2apply)
    train_loader, validation_loader = create_datasets_split(dataset, True, 0.8, 256, 128)
    fcnet = fullyconnected().to(device)

    print("training...")
    train_fully_connected(opt, b_list, motion_list, feat_extract, train_loader, validation_loader, 30, 0.001, device,
                          fcnet)

# if __name__ == "__main__":
#     opt = get_args()
#     print(opt)
#     train(opt)
