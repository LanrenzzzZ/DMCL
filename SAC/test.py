import os
import time
import numpy as np
import torch
import torch.nn as nn
from velodyne_env import GazeboEnv

from utils import IOStream
import argparse
from Ctmr_sac import make_agent
from Replay_Buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from Data_Augmentation import center_crop_image
# /home/jjh/DRL-robot-navigation/TD3
dirPath = os.path.dirname(os.path.realpath(__file__))

def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action

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

def evaluate(eval_episodes=10, encoder_type='Pixel', image_size=84,sample_stochastically=False):
    all_ep_rewards = []
    collison = 0
    success = 0
    for i in range(eval_episodes):
        state, obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done and step < 501:
            if encoder_type == 'pixel':
                obs = center_crop_image(obs, image_size)

            with eval_mode(agent):
                if sample_stochastically:
                    action = agent.sample_action(state, obs)
                    unnorm_action = np.array([
                        action_unnormalized(action[0], ACTION_V_MAX, 0),
                        action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)
                    ])
                else:
                    action = agent.select_action(state, obs)
                    unnorm_action = np.array([
                        action_unnormalized(action[0], ACTION_V_MAX, 0),
                        action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)
                    ])
                state, obs, reward, done, target = env.step(unnorm_action)
                episode_reward += reward
                step += 1
            if reward < -90:
                collison += 1
            if target:
                success += 1
        all_ep_rewards.append(episode_reward)

    mean_ep_reward = np.mean(all_ep_rewards)
    best_ep_reward = np.max(all_ep_rewards)
    avg_col = collison / eval_episodes
    avg_success = success / eval_episodes
    io.cprint("..............................................")
    io.cprint("Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f, %f" % (
    eval_episodes, epoch, mean_ep_reward, avg_col, avg_success))
    io.cprint('Evaluation  Mean :%f Best : %f' % (mean_ep_reward, best_ep_reward))
    io.cprint("..............................................")
    return mean_ep_reward, best_ep_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--env_name", default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int, help="After how many steps to perform the evaluation 5e3")
    parser.add_argument("--max_ep", default=500, type=int)
    parser.add_argument("--max_timesteps", default=5e6, type=int, help="Maximum number of steps to perform")
    parser.add_argument("--random_near_obstacle", default=True, type=bool,
                        help="To take random actions near obstacles or not")
    parser.add_argument('--image_size', default=84, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--goal_dim', default=4, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--init_steps', default=1000, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_attn_layer', default=1, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)  # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int)  # try to change it to 1 and retain 0.01 above
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)

    parser.add_argument('--actor_attach_encoder', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--actor_coeff', default=1., type=float)
    parser.add_argument('--adam_warmup_step', type=float)
    parser.add_argument('--mtm_length', default=20, type=int)
    parser.add_argument('--encoder_annealling', default=False, action='store_true')
    parser.add_argument('--mtm_bsz', default=64, type=int)
    parser.add_argument('--mtm_ratio', default=0.25, type=float)
    parser.add_argument('--mtm_not_ema', default=False, action='store_true')
    parser.add_argument('--normalize_before', default=False, action='store_true')
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--attention_dropout', default=0., type=float)
    parser.add_argument('--relu_dropout', default=0., type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
    # Create the network storage folders
    if not os.path.exists("./results"):
        os.makedirs("./results")

    io = IOStream('./results/output' + '/output_mlr_0.7_16_1.log')
    env = GazeboEnv('multi_robot_scenario.launch', 1, 1, 1)
    time.sleep(5)
    # env.seed(seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    action_shape = 2
    obs_shape = (3, 84, 84)
    pre_aug_obs_shape = (3, 100, 100)
    replay_buffer_size = 100000
    laser_shape = 24
    episode, episode_reward = 0, 0
    save_model_replay = False

    # Create a replay buffer
    replay_buffer = ReplayBuffer(
        laser_shape=laser_shape,
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        mtm_bsz=args.mtm_bsz,
        mtm_length=args.mtm_length,
        mtm_ratio=args.mtm_ratio
    )

    ACTION_V_MIN = 0    # m/s
    ACTION_W_MIN = -1  # rad/s
    ACTION_V_MAX = 1  # m/s
    ACTION_W_MAX = 1  # rad/s
    # Create the network
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )
    agent.load(dirPath, 27)
    # Create evaluation data store
    evaluations = []
    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1
    frame_step = 3
    count_rand_actions = 0
    random_action = []
    writer = SummaryWriter(dirPath + '/evaluations/', flush_secs=1, max_queue=1)
    # Begin the training loop

    while True:
        mean, best = evaluate(eval_episodes=500)

