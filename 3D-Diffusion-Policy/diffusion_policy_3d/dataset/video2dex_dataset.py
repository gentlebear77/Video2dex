from typing import Dict
import torch
import numpy as np
import copy
import pathlib
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
import pickle

class VideoDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            max_train_episodes=None,
            val_ratio=0.0,
            task_name=None,
        ):
        super().__init__()
        self.task_name = task_name
        data_directory = pathlib.Path(zarr_path)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        with open(data_directory,'rb') as file:
            demo_paths = list(pickle.load(file).values())
        for demo_part in demo_paths:
            observations = demo_part["observations"]
            actions = demo_part["actions"]
            data = {
                'obs' : observations,
                'action' : actions
            }
            self.replay_buffer.add_episode(data)
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'obs': self.replay_buffer['obs'],
            'action': self.replay_buffer['action']
        }
        if 'range_eps' not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs['range_eps'] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
