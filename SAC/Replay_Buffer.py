import os
from collections import deque
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from Data_Augmentation import center_crop_image, random_crop, random_crop_2


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(self, laser_shape, obs_shape, action_shape, capacity, batch_size, device, image_size=84, transform=None,
                 mtm_bsz=64,
                 mtm_length=10,
                 mtm_ratio=0.15
                 ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        self.mtm_ratio = mtm_ratio
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.laser = np.empty((capacity, laser_shape), dtype=np.float32)  # (100000, 24)
        self.next_laser = np.empty((capacity, laser_shape), dtype=np.float32)  # (100000, 24)
        self.obses = np.empty((capacity, obs_shape[0], obs_shape[1], obs_shape[2]), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, obs_shape[0], obs_shape[1], obs_shape[2]), dtype=obs_dtype)
        self.actions = np.empty((capacity, action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.mtm_bsz = mtm_bsz
        self.mtm_length = mtm_length

    def add(self, laser, obs, action, reward, next_laser, next_obs, done):

        np.copyto(self.laser[self.idx], laser)
        np.copyto(self.next_laser[self.idx], next_laser)
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        laser = self.laser[idxs]
        next_laser = self.next_laser[idxs]
        obses = self.obses[idxs] # (64, 3, 140, 140)
        next_obses = self.next_obses[idxs]
        mid_obses = obses[:, :, 20:120, 20:120]
        mid_obses = random_crop(mid_obses, self.image_size)
        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)

        laser = torch.as_tensor(laser, device=self.device).float()
        next_laser = torch.as_tensor(next_laser, device=self.device).float()
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()

        mid_obses = torch.as_tensor(mid_obses, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return laser, obses, actions, rewards, next_laser, next_obses, mid_obses, not_dones

    def sample_ctmr(self):
        idxs = np.random.randint(
            0, self.capacity - self.mtm_length if self.full else self.idx - self.mtm_length, size=self.mtm_bsz)
        idxs = idxs.reshape(-1, 1)
        step = np.arange(self.mtm_length).reshape(1, -1)
        idxs = idxs + step
        obses_label = self.obses[idxs]
        actions_label = self.actions[idxs]
        non_masked = np.zeros((self.mtm_bsz, self.mtm_length), dtype=np.bool)

        obses = self.random_obs(obses_label, non_masked)
        non_masked = torch.as_tensor(non_masked, device=self.device)
        obses_label = random_crop_2(obses_label, self.image_size)
        obses = random_crop_2(obses, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        obses_label = torch.as_tensor(obses_label, device=self.device).float()
        actions_label = torch.as_tensor(actions_label, device=self.device).float()
        return (*self.sample_cpc(), dict(obses=obses, obses_label=obses_label,actions_label=actions_label,
                                         non_masked=non_masked))

    def sample_curl(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def random_obs(self, obses, non_masked):
        masked_obses = np.array(obses, copy=True)
        for row in range(self.mtm_bsz):
            for col in range(self.mtm_length):
                prob = random.random()
                if prob < self.mtm_ratio:
                    prob /= self.mtm_ratio
                    if prob < 0.8:
                        masked_obses[row, col] = 0
                    elif prob < 0.9:
                        masked_obses[row, col] = self.get_random_obs()
                    non_masked[row, col] = True
        return masked_obses

    def random_act(self, actions, non_masked):
        masked_actions = np.array(actions, copy=True)
        for row in range(self.mtm_bsz):
            for col in range(self.mtm_length):
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    if prob < 0.8:
                        masked_actions[row, col] = 0
                    elif prob < 0.9:
                        masked_actions[row, col] = self.get_random_act()
                    non_masked[row, col + self.mtm_length + 1] = True
        return masked_actions

    def get_random_act(self):
        idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        return self.actions[idx]

    def get_random_obs(self):
        idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        obs = self.obses[idx]
        return obs

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity
