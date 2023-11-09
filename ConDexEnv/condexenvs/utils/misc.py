import os 
import time
import numpy as np
from collections import deque

class RewardNormalizer(object):
    def __init__(self, is_norm, update_freq=10, name='default'):
        self.reward_mean = 0
        self.reward_std = 1
        self.num_steps = 0
        self.vk = 0
        self.is_norm = is_norm
        ''' to log running mu,std '''
        # self.writer = writer
        self.update_freq = update_freq
        self.name = name
    
    def update_mean_std(self, reward):
        self.num_steps += 1
        if self.num_steps == 1:
            # the first step, no need to normalize
            self.reward_mean = reward
            self.vk = 0
            self.reward_std = 1
        else:
            # running mean, running std
            delt = reward - self.reward_mean
            self.reward_mean = self.reward_mean + delt/self.num_steps
            self.vk = self.vk + delt * (reward-self.reward_mean)
            self.reward_std = np.sqrt(self.vk/(self.num_steps - 1))
    
    def get_normalized_reward(self, rewards):
        rewards_norm = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        return rewards_norm
    
    # def update_writer(self):
    #     self.writer.add_scalar(f'Episode_rewards/RunningMean_{self.name}', np.mean(self.reward_mean), self.num_steps)
    #     self.writer.add_scalar(f'Episode_rewards/RunningStd_{self.name}', np.mean(self.reward_std), self.num_steps)

    def update(self, reward, is_eval=False):
        if not is_eval and self.is_norm:
            if type(reward) is np.ndarray:
                for item in reward:
                    self.update_mean_std(item)
            else:
                self.update_mean_std(reward)
            reward = self.get_normalized_reward(reward)
            ''' log the running mean/std '''
            # if self.num_steps % self.update_freq == 0:
            #     self.update_writer()
        return reward