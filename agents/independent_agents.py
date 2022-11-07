import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions import Normal, Categorical
from common.models import *
from torchmetrics.functional import accuracy
import numpy as np

class Communicators(pl.LightningModule):
    def __init__(self, m_dim=10, lr=0.00003, noise_variance=0, method='supervised_learning',
                 discrete=False, speaker_lr=0.0003, listener_lr=0.0003, ps_weight=0.1,
                 pl_weight=0.01, speaker_ent_target=1.0, speaker_ent_bonus=0.0,
                 speaker_lambda=0.3, listener_ent_bonus=0.03, ce_loss=0.001,
                 gradient_clipping=1.0, ps_entropy=1.0, dropout=0.0, weight_decay=0.0, ps_ent_switch=50):
        super(Communicators, self).__init__()
        self.save_hyperparameters()
        self.speaker = Speaker(m_dim=self.hparams.m_dim, discrete=self.hparams.discrete,
                               end_to_end=False, dropout=self.hparams.dropout)
        self.listener = Listener(m_dim=self.hparams.m_dim)
        self.noise_variance = noise_variance
        self.automatic_optimization = False  # More control facilitating the independent agents
        if self.noise_variance == 0:
            self.noisy_channel = False
        else:
            self.noisy_channel = True
            self.noise = Normal(0.0, self.noise_variance)
        # TODO: Figure out how to do this stuff tomorrow, I'm too tired.
        if self.hparams.ps_entropy == 'linear':
            self.ps_ent_val = lambda x: max([1-(x/self.hparams.ps_ent_switch), 0])
        elif self.hparams.ps_entropy == 'tanh':
            #self.ps_ent_val = lambda x, s, c_p: -np.tanh(0.1*(x-50))
            self.ps_ent_val = lambda x: -np.tanh(0.1*(x-self.hparams.ps_ent_switch))
        elif self.hparams.ps_entropy == 'const':
            self.ps_ent_val = lambda x: self.hparams.ps_ent_switch
        elif self.hparams.ps_entropy == 'step':
            self.ps_ent_val = lambda x: 1 if x < self.hparams.ps_ent_switch else 0
        else:
            self.ps_ent_val = lambda x: 1.0


    def forward(self, im1, im2, custom_message=False, m_custom=None):
        """
        In some instances, I want a custom message to see how the agent interprets the message.
        In that instance, I don't want noise to corrupt the message.
        TODO: Insert messaging sampling
        :param im1:
        :param im2:
        :param custom_message:
        :param m_custom:
        :return: logits, transmitted message, corrupted message
        """
        m_dist = None  # In the case of the custom message, this won't be instantiated in first loop
        m = None
        if not custom_message:
            s_logits = self.speaker(im1)
            m_dist = Categorical(logits=s_logits)
            m = m_dist.sample()
        else:
            m = m_custom
        if self.noisy_channel and not custom_message:
            m_corrupt = F.one_hot(m.detach(), num_classes=self.hparams.m_dim) + self.noise.sample(m.shape).to(self.device)
        else:
            m_corrupt = F.one_hot(m.detach(), num_classes=self.hparams.m_dim)
        output = self.listener(im2, m_corrupt)
        return output, m, m_corrupt, m_dist

    def positive_listening(self, dist, image):
        """
        This method intends to measure positive listening, w/r to the speaker.
        The speaker policy shall be evaluated empirically.
        :return: Causal Influence
        """
        batch_size = dist.probs.shape[0]
        no_m_logits = self.listener(image, torch.zeros(batch_size, self.hparams.m_dim).to(self.device))
        dist_no_m = Categorical(logits=no_m_logits)
        l1_norm = -torch.norm(dist.probs-dist_no_m.probs.detach(), p=1, dim=1).sum()
        ce = -dist.probs.detach() * dist_no_m.logits
        loss_ce = ce.sum(dim=1).sum()
        return (self.hparams.pl_weight * l1_norm) + (self.hparams.ce_loss * loss_ce)

    def positive_signalling(self, dist):
        """
        This method intends to measure the mutual information between an image and the message.
        Shall have to find a way to do this empirically.
        :return: Mutual Information
        """
        batch_size = dist.probs.shape[0]
        mpol = dist.probs.mean(dim=0)
        mpol_ent = Categorical(mpol).entropy()
        cond_ent = torch.pow(dist.entropy() - self.hparams.speaker_ent_target, 2).sum()
        ps_entropy = self.ps_ent_val(self.current_epoch)
        ps = -((batch_size*ps_entropy*mpol_ent) - (self.hparams.speaker_lambda * cond_ent))
        return self.hparams.ps_weight * ps

    def configure_optimizers(self):
        speaker_optimizer = torch.optim.Adam(self.speaker.parameters(), lr=self.hparams.speaker_lr,
                                             weight_decay=self.hparams.weight_decay)
        listener_optimizer = torch.optim.Adam(self.listener.parameters(), lr=self.hparams.listener_lr)
        return speaker_optimizer, listener_optimizer

    def training_step(self, batch, batch_idx):
        s_opt, l_opt = self.optimizers()
        s_opt.zero_grad()
        l_opt.zero_grad()
        x1, y1 = batch['speaker']
        x2, y2 = batch['listener']
        data, answers, dist, m, m_corrupt, s_dist = self.forward_pass(x1, x2, y1, y2)
        # Speaker Losses
        self.manual_backward(data['speaker_loss'])
        torch.nn.utils.clip_grad_norm_(self.speaker.parameters(), self.hparams.gradient_clipping)
        s_opt.step()
        # Listener Losses
        self.manual_backward(data['listener_loss'])
        torch.nn.utils.clip_grad_norm_(self.listener.parameters(), self.hparams.gradient_clipping)
        l_opt.step()
        self.log('training', data)
        return data['listener_loss']

    @staticmethod
    def sl_objective(pred, target):
        loss = F.cross_entropy(pred, target)
        return loss

    def rl_objective_speaker(self, distribution, answers, reward):
        loss = -torch.sum(distribution.log_prob(answers) * reward) - \
               (self.hparams.speaker_ent_bonus * distribution.entropy().mean()) +\
               self.positive_signalling(distribution)
        return loss

    def rl_objective_listener(self, distribution, answers, reward, image):
        loss = -torch.sum(distribution.log_prob(answers) * reward) - \
               (self.hparams.listener_ent_bonus * distribution.entropy().mean()) +\
               self.positive_listening(distribution, image)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, y1 = batch['speaker']
        x2, y2 = batch['listener']
        data, answers, dist, m, m_corrupt, s_dist = self.forward_pass(x1, x2, y1, y2)
        self.log('validation', data)
        return answers, m, m_corrupt

    def test_step(self, batch, batch_idx):
        x1, y1 = batch['speaker']
        x2, y2 = batch['listener']
        data, answers, dist, m, m_corrupt, s_dist = self.forward_pass(x1, x2, y1, y2)
        self.log('test_full', data)
        return answers, F.one_hot(m, num_classes=self.hparams.m_dim), m_corrupt

    def predict_step(self, batch, batch_idx):
        x1, y1 = batch['speaker']
        x2, y2 = batch['listener']
        data, answers, dist, m, m_corrupt, s_dist = self.forward_pass(x1, x2, y1, y2)
        self.log('predict', {'message': m.mean()})
        return answers

    def calculate_metrics(self, logits, y1, y2):
        dist = self.convert_distribution(logits)
        answer = dist.sample()
        reward = -torch.ones_like(y1).type(torch.float32)
        indexes = torch.where(answer == (y1+y2))[0]
        reward[indexes] = 1.0
        return reward, answer, dist

    def forward_pass(self, x1, x2, y1, y2):
        y_hat, m, m_corrupt, s_dist = self(x1, x2)
        reward, answers, l_dist = self.calculate_metrics(y_hat, y1, y2)
        if self.hparams.method == 'supervised_learning':  # Note I can't do SL end to end in this regime. So RL.
            speaker_loss = self.rl_objective_speaker(s_dist, m, reward)
            listener_loss = self.sl_objective(y_hat, y1+y2)
        elif self.hparams.method == 'reinforcement_learning':
            speaker_loss = self.rl_objective_speaker(s_dist, m, reward)
            listener_loss = self.rl_objective_listener(l_dist, answers, reward, x2)
        else:
            raise NotImplementedError
        return {"speaker_loss": speaker_loss, "reward": reward.mean(), "speaker_entropy": s_dist.entropy().mean(),
                "listener_loss": listener_loss, "listener_entropy": l_dist.entropy().mean()}, \
                 answers, l_dist, m, m_corrupt, s_dist

    @staticmethod
    def convert_distribution(logits):
        return Categorical(logits=logits)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Communicators")
        parser.add_argument("--speaker_lr", type=float, default=0.0003)
        parser.add_argument("--listener_lr", type=float, default=0.0003)
        parser.add_argument("--ps_weight", type=float, default=0.1)
        parser.add_argument("--pl_weight", type=float, default=0.01)
        parser.add_argument("--speaker_ent_target", type=float, default=1.0)
        parser.add_argument("--speaker_ent_bonus", type=float, default=0.0)
        parser.add_argument("--speaker_lambda", type=float, default=0.3)
        parser.add_argument("--listener_ent_bonus", type=float, default=0.03)
        parser.add_argument("--ce_loss", type=float, default=0.001)
        return parent_parser




