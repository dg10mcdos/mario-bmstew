import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import torch

import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil, csv, time
from BBRL.src.helpers import flag_get
from BBRL.src.RLNet import RLNet
from BBRL.src.env import MultipleEnvironments
from BBRL.src.process import evaluate
from BBRL.src.model import PPO
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


def check_flag(info):
    out = 0
    for i in info:
        if flag_get(i):
            out += 1
    return out


def train(b_list,motion_list):  # opt is object storing args
    opt = get_args()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    opt.saved_path = os.getcwd() + '/BBRL/' + opt.saved_path

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    savefile = opt.saved_path + '/BBRL_train.csv'
    title = ['Loops', 'Steps', 'Time', 'AvgLoss', 'MeanReward', "StdReward", "TotalReward", "Flags"]
    with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

    # Create environments
    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)

    model = PPO(envs.num_states, envs.num_actions) # 4 states(assuming processes), 7 actions (buttons)
    # model = RLNet(envs.num_states, envs.num_actions)

    # Create model and optimizer
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()  # pytorch what it says on the tin.
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if TEST_ON_THE_GO:
        # evaluate(opt, model, envs.num_states, envs.num_actions)
        mp = _mp.get_context("spawn")
        process = mp.Process(target=evaluate,
                             args=(opt, model, envs.num_states, envs.num_actions))  # 4 states, 7 actions (buttons)
        process.start()
    # Reset envs
    curr_states = []  # current state of each env
    [curr_states.append(env.reset()) for env in envs.envs]

    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()  # chuck it on gpu

    tot_loops = 0
    tot_steps = 0

    # Start main loop (EXECUTED AFTER EACH ACTION)
    while True:
        # Save model each loop
        # if tot_loops % opt.save_interval == 0 and tot_loops > 0:
        #     # torch.save(model.state_dict(), "{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
        #     torch.save(model.state_dict(), "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, tot_loops))

        start_time = time.time()

        # Accumulate evidence
        tot_loops += 1
        old_log_policies = []
        actions = []  # record of actions
        values = []  # record of q values
        states = []  # record of states of processes
        rewards = []
        dones = []
        flags = []
        for _ in range(opt.num_local_steps):
            # From given states, predict an action
            states.append(curr_states)

            logits, value = model(curr_states)  # actor, critic

            # actor [4,7] "score" for pressing buttons aka given our state probability of each action
            # critic [4,1] q-value for state
            values.append(value.squeeze())  # critic [4,1] -> 4
            policy = F.softmax(logits, dim=1)  # turns action scores into probabilities
            old_m = Categorical(policy)  # actions with probabilities
            action = old_m.sample()  # choose random action wrt probabilities
            actions.append(action)  # record action
            old_log_policy = old_m.log_prob(action)  # probability of action
            old_log_policies.append(old_log_policy)  # record action probability

            # Evaluate predicted action
            result = []
            # ac = action.cpu().item()
            if torch.cuda.is_available():
                # [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
                [result.append(env.step(act.item())) for env, act in zip(envs.envs, action.cpu())]  # take action,
            else:
                # [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]
                [result.append(env.step(act.item())) for env, act in zip(envs.envs, action)]
                # result :
                # state[array(28224)],
                # reward - number
                # done - boolean
                # info - [{'levelLo': 0, 'enemyType5': 0, 'xscrollLo': 16, 'floatState': 0, 'enemyType4': 0, 'status': 8, 'levelHi': 0, 'score': 0, 'lives': 2,
                # 'scrollamaount': 1, 'scrolling': 16, 'xscrollHi': 0, 'enemyType2': 0, 'time': 397, 'enemyType3': 0, 'coins': 0, 'gameMode': 1, 'enemyType1': 0}]
            state, reward, done, info = zip(*result)
            state = torch.from_numpy(np.concatenate(state, 0))

            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)

            rewards.append(reward)
            dones.append(done)
            flags.append(check_flag(info) / opt.num_processes)
            curr_states = state

        # Training stage
        _, next_value, = model(curr_states)  # retrieve next q value
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
                logits, value = model(states[batch_indices])
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


if __name__ == "__main__":
    opt = get_args()
    train(opt)
