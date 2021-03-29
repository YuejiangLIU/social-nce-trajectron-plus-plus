import math
import torch

class EventSampler():
    '''
    Different sampling strategies for social contrastive learning
    '''

    def __init__(self, device='cuda'):
        # fixed param
        self.noise_local = 0.02
        self.min_seperation = 0.2                                   # env-dependent parameter, diameter of agents
        self.max_seperation = 2.5                                   # env-dependent parameter, diameter of agents
        self.agent_zone = self.min_seperation * torch.tensor([
            [1.0, 0.0], [-1.0, 0.0],
            [0.0, 1.0], [0.0, -1.0],
            [0.707, 0.707], [0.707, -0.707],
            [-0.707, 0.707], [-0.707, -0.707]], device=device)      # regional surroundings
        self.device = device

    def _valid_check(self, pos_seed, neg_seed):
        '''
        Check validity of sample seeds, mask out the frames that are invalid due to nan
        '''
        dist = (neg_seed - pos_seed.unsqueeze(1)).norm(dim=-1)
        mask_valid = (dist > self.min_seperation) & (dist < self.max_seperation)

        dmin = torch.where(torch.isnan(dist[mask_valid]), torch.full_like(dist[mask_valid], 1000.0), dist[mask_valid]).min()
        assert dmin > self.min_seperation

        return mask_valid.unsqueeze(-1)

    def social_sampling(self, primary_curr, primary_next, neighbors_next):
        '''
        Draw negative samples based on regions of other agents in the future
        '''

        # positive
        sample_pos = primary_next[..., :2] - primary_curr[..., :2]
        sample_pos += torch.rand(sample_pos.size(), device=self.device).sub(0.5) * self.noise_local

        # neighbor territory
        sample_neg = neighbors_next[..., :2] - primary_curr[..., None, :2]
        sample_neg = sample_neg.unsqueeze(2) + self.agent_zone[None, None, :, None, :]
        sample_neg = sample_neg.view(sample_neg.size(0), sample_neg.size(1) * sample_neg.size(2), sample_neg.size(3), 2)
        sample_neg += torch.rand(sample_neg.size(), device=self.device).sub(0.5) * self.noise_local

        mask_valid = self._valid_check(sample_pos, sample_neg)
        sample_neg.masked_fill_(~mask_valid, float('nan'))

        return sample_pos, sample_neg, mask_valid

    def local_sampling(self, primary_curr, primary_next, neighbors_next):
        '''
        Draw negative samples centered around the primary agent in the future
        '''

        # positive
        sample_pos = primary_next[..., :2] - primary_curr[..., :2]
        sample_pos += torch.rand(sample_pos.size(), device=self.device).sub(0.5) * self.noise_local

        # neighbor territory
        radius = torch.rand(sample_pos.size(0), 16, device=self.device) * self.max_seperation * 0.8 + self.max_seperation * 0.2
        theta = torch.rand(sample_pos.size(0), 16, device=self.device) * 2 * math.pi
        dx = radius * torch.cos(theta)
        dy = radius * torch.sin(theta)

        sample_neg = torch.stack([dx, dy], axis=2).unsqueeze(axis=2) + sample_pos.unsqueeze(1)

        mask_valid = self._valid_check(sample_pos, sample_neg)
        sample_neg.masked_fill_(~mask_valid, float('nan'))

        return sample_pos, sample_neg, mask_valid
