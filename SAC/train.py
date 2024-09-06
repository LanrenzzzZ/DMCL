import time
import rospy
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from velodyne_env import GazeboEnv
from Data_Augmentation import center_crop_image, random_crop
from Ctmr_sac import make_agent
from Replay_Buffer import ReplayBuffer
import argparse
from utils import IOStream

dirPath = os.path.dirname(os.path.realpath(__file__))
#----------------------------------------------------------
ACTION_V_MIN = 0.0 # m/s
ACTION_W_MIN = -0.5 # rad/s
ACTION_V_MAX = 0.20 # m/s
ACTION_W_MAX = 0.5 # rad/s
#****************************
is_training = True
#----------------------------------------
def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action
#**********************************

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def evaluate(num_epsiodes=3, encoder_type='pixel', image_size=84, sample_stochastically=True):
    all_ep_rewards = []

    for i in range(num_epsiodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done:
            if encoder_type == 'pixel':
                obs = center_crop_image(obs, image_size)
            with eval_mode(agent):
                if sample_stochastically:
                    action = agent.sample_action(obs)
                    unnorm_action = np.array([
                        action_unnormalized(action[0], ACTION_V_MAX, 0),
                        action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)
                    ])
                else:
                    action = agent.select_action(obs)
                    unnorm_action = np.array([
                    action_unnormalized(action[0], ACTION_V_MAX, 0),
                    action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)
                ])

            obs, reward, done = env.step(unnorm_action)
            episode_reward += reward
            step +=1
            if step > 750:
                print('final', i, step)
                done = True
                break
        all_ep_rewards.append(episode_reward)

    mean_ep_reward = np.mean(all_ep_rewards)
    best_ep_reward = np.max(all_ep_rewards)

    print('Evaluation: Mean, Best')
    print(mean_ep_reward, best_ep_reward)

    return mean_ep_reward, best_ep_reward

if __name__ == "__mian__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/output")
    parser.add_argument("--env_name", default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int, help="After how many steps to perform the evaluation 5e3")
    parser.add_argument("--max_ep", default=500, type=int)
    parser.add_argument("--start_steps", default=2000, type=int, help='Number of steps for the warm-up stage using random policy')
    parser.add_argument("--eval_ep", default=10, type=int, help="number of episodes for evaluation")
    parser.add_argument("--max_timesteps", default=5e6, type=int, help="Maximum number of steps to perform")
    parser.add_argument("--save_models", default=True, type=bool, help="Weather to save the model or not")
    parser.add_argument("--expl_noise", default=1, type=int,
                        help="Initial exploration noise starting value in range [expl_min ... 1]")
    parser.add_argument("--expl_decay_steps", default=500000, type=int,
                        help="Number of steps over which the initial exploration noise will decay over")
    parser.add_argument("--expl_min", default=0.1, type=float,
                        help="Exploration noise after the decay in range [0...expl_noise]")
    parser.add_argument("--batch_size", default=1, type=int, help="# Size of the mini-batch")
    parser.add_argument("--discount", default=0.99999, type=float,
                        help="Discount factor to calculate the discounted future reward (should be close to 1)")
    parser.add_argument("--tau", default=0.005, type=float, help="Soft target update variable (should be close to 0)")
    parser.add_argument("--policy_noise", default=0.2, type=float, help="Added noise for exploration")
    parser.add_argument("--noise_clip", default=0.5, type=float, help="Maximum clamping values of the noise")
    parser.add_argument("--policy_freq", default=2, type=int, help="Frequency of Actor network updates")
    parser.add_argument("--buffer_size", default=1e6, type=int, help="Maximum size of the buffer")
    parser.add_argument("--file_name", default='TD3_velodyne', type=str, help="name of the file to store the policy")
    parser.add_argument("--random_near_obstacle", default=True, type=bool,
                        help="To take random actions near obstacles or not")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    io = IOStream('./results/output' + '/output_tf_2_256.log')

    env = GazeboEnv('multi_robot_scenario.launch', 1, 1, 1)
    time.sleep(5)
    # env.seed(seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    action_shape = 2
    obs_shape = (3, 84, 84)
    pre_aug_obs_shape = (3, 100, 100)
    replay_buffer_size = 100000
    episode, episode_reward, done = 0, 0, True
    max_steps = 1000000
    # initial_step = 0
    initial_step = 25000
    save_model_replay = False


    replay_buffer = ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=replay_buffer_size,
        batch_size=128,
        device=device,
        image_size=84,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device
    )

    # agent.load(dirPath, initial_step)   # 5 + 149 + 30
    # replay_buffer.load(dirPath + '/replay_memory/')
    writer = SummaryWriter(dirPath + '/evaluations/', flush_secs=1, max_queue=1)

    for step in range(initial_step, max_steps):
        if step % 1000 == 0:
            print('start_eval')
            mean, best = evaluate()
            writer.add_scalar('Reward mean', mean, step)
            writer.add_scalar('Reward best', best, step)
            if save_model_replay:
                if step % (1000) == 0:
                    agent.save(dirPath, step)
                    agent.save_curl(dirPath, step)
                    replay_buffer.save(dirPath + '/replay_memory')
                    print('saved model and replay memory', step)

                    done = True
                    save_model_replay = True
        if done:
            print("*********************************")
            print('Episode: ' + str(episode) + ' training')
            print('Step: ' + str(step) + ' training')
            print('Reward average per ep: ' + str(episode_reward))
            print("*********************************")

            obs = env.reset()
            done = False
            episode_reward = 0
            episode += 1

        if step < 200:
            action = np.array([
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
            ])
            unnorm_action = np.array([
                action_unnormalized(action[0], ACTION_V_MAX, 0),
                action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)
            ])


        else:
            with eval_mode(agent):
                action = agent.sample_action(obs)
                unnorm_action = np.array([
                    action_unnormalized(action[0], ACTION_V_MAX, 0),
                    action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)
                    ])

        if step >= (initial_step + 400): # 1000
            num_updates =1
            for _ in range(num_updates):
                agent.update(replay_buffer, step)

        next_obs, reward, done = env.step(unnorm_action)


        # print('reward ', reward)
        # print('state ', obs.shape)

        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done)
        if reward < -1.:
            print('\n----collide-----\n')
            for _ in range(1):
                # print('aqui2')
                replay_buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs
        # episode_step += 1














