import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from snce.sampling import EventSampler

import matplotlib
matplotlib.use('Agg')


class SocialNCE():
    '''
    Social contrastive loss, encourage the extracted motion representation to be aware of socially unacceptable events
    '''

    def __init__(self, head_projection=None, encoder_sample=None, sampling='social', horizon=3, temperature=0.1):
        # encoders
        self.head_projection = head_projection
        self.encoder_sample = encoder_sample
        # nce
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        # sampling
        self.sampling = sampling
        self.horizon = horizon
        self.sampler = EventSampler()

    def loss(self, primary_prev, neighbors_prev, primary_next, neighbors_next, feat):
        '''
        Social NCE Loss
        '''

        # sampling
        if self.sampling == 'social':
            sample_pos, sample_neg, mask_valid = self.sampler.social_sampling(
                primary_prev[:, -1:, ], primary_next, neighbors_next)
        elif self.sampling == 'local':
            sample_pos, sample_neg, mask_valid = self.sampler.local_sampling(
                primary_prev[:, -1:, ], primary_next, neighbors_next)
        else:
            raise NotImplementedError

        # self._sanity_check(primary_prev, neighbors_prev, primary_next, neighbors_next, sample_pos, sample_neg, mask_valid)
        # pdb.set_trace()

        # nan pre-process: set nan to 0 in forward to ensure grad
        sample_neg.masked_fill_(~mask_valid, 0.0)

        candidate_pos = sample_pos[:, 1:self.horizon+1]
        candidate_neg = sample_neg[:, :, 1:self.horizon+1]

        # temporal
        time_pos = (torch.ones(candidate_pos.size(0))[:, None] * (torch.arange(self.horizon) - (
            self.horizon-1.0)*(0.5))[None, :]).to(candidate_pos.device) / self.horizon
        time_neg = (torch.ones(candidate_neg.size(0), candidate_neg.size(1))[:, :, None] * (torch.arange(
            self.horizon) - (self.horizon-1.0)*(0.5))[None, None, :]).to(candidate_neg.device) / self.horizon

        # embedding
        emb_obsv = self.head_projection(feat[:, :, :1])
        emb_pos = self.encoder_sample(candidate_pos, time_pos[:, :, None])
        emb_neg = self.encoder_sample(candidate_neg, time_neg[:, :, :, None])

        # normalization
        query = nn.functional.normalize(emb_obsv, dim=-1)
        key_pos = nn.functional.normalize(emb_pos, dim=-1)
        key_neg = nn.functional.normalize(emb_neg, dim=-1)

        # similarity
        sim_pos = (query * key_pos.unsqueeze(1)).sum(dim=-1)
        sim_neg = (query.unsqueeze(2) * key_neg.unsqueeze(1)).sum(dim=-1)

        # nan post-process: set nan negatives to large negative value
        sim_neg.masked_fill_(
            ~mask_valid[:, None, :, 1:self.horizon+1, 0], -10.0)

        # logits
        sim_pos_avg = sim_pos.mean(axis=1)              # average over samples
        sz_neg = sim_neg.size()
        sim_neg_flat = sim_neg.view(sz_neg[0], sz_neg[1]*sz_neg[2], sz_neg[3])
        logits = torch.cat([sim_pos_avg.view(sz_neg[0]*sz_neg[3], 1), sim_neg_flat.view(
            sz_neg[0]*sz_neg[3], sz_neg[1]*sz_neg[2])], dim=1) / self.temperature

        # loss
        labels = torch.zeros(logits.size(
            0), dtype=torch.long, device=logits.device)
        loss = self.criterion(logits, labels)

        return loss

    def _sanity_check(self, primary_prev, neighbors_prev, primary_next, neighbors_next, sample_pos, sample_neg, mask_valid):
        '''
        Check sampling strategy
        '''
        for i in range(40):
            for k in range(1, self.horizon):
                sample_pos_raw = primary_prev[i, -1, :2] + sample_pos[i, k]
                sample_neg_raw = primary_prev[i, -1,
                                              :2].unsqueeze(0) + sample_neg[i, :, k]
                sample_neg_raw = sample_neg_raw[mask_valid[i, :, k].squeeze()]
                if len(sample_neg_raw) > 0:
                    self._visualize_samples(primary_prev[i, :, :2].cpu().numpy(),
                                            neighbors_prev[i, ..., :2].cpu().numpy(),
                                            primary_next[i, :, :2].cpu().numpy(),
                                            neighbors_next[i, ..., :2].cpu().numpy(),
                                            sample_pos_raw.cpu().numpy(), sample_neg_raw.cpu().numpy(),
                                            fname='sanity/samples_{:d}_time_{:d}.png'.format(i, k))

    def _visualize_samples(self, primary_prev_frame, neighbors_prev_frame, primary_next_frame, neighbors_next_frame, sample_pos_frame, sample_neg_frame, fname, window=4.0):

        fig = plt.figure(frameon=False)
        fig.set_size_inches(8, 6)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(primary_prev_frame[:, 0],
                primary_prev_frame[:, 1], 'k-o', markersize=4)
        ax.plot(primary_next_frame[:, 0], primary_next_frame[:, 1], 'k-.')

        for i in range(neighbors_prev_frame.shape[0]):
            ax.plot(
                neighbors_prev_frame[i, :, 0], neighbors_prev_frame[i, :, 1], 'c-o', markersize=4)
            ax.plot(neighbors_next_frame[i, :, 0],
                    neighbors_next_frame[i, :, 1], 'c-.')

        ax.plot(sample_pos_frame[0], sample_pos_frame[1], 'gs')
        ax.plot(sample_neg_frame[:, 0], sample_neg_frame[:, 1], 'rx')

        ax.set_xlim(primary_prev_frame[-1, 0] - window, primary_prev_frame[-1, 0] + window)
        ax.set_ylim(primary_prev_frame[-1, 1] - window, primary_prev_frame[-1, 1] + window)
        ax.set_aspect('equal')
        plt.grid()

        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
