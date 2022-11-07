import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions import Normal, Categorical


class MNISTconv(pl.LightningModule):

    def __init__(self):
        super(MNISTconv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.conv2 = nn.Conv2d(32, 64, 5, 1) # I changed this helloooooooo 5 #TODO required changing next layer to 1600
        self.pool = nn.MaxPool2d(2, 2)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.flatten(1)


class SpeakerNet(pl.LightningModule):

    def __init__(self, m_dim=10, discrete=False, end_to_end=False, dropout=0.0):
        super(SpeakerNet, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        #self.fc2 = nn.Linear(1024, 512)
        #nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.message = nn.Linear(1024, m_dim)  # this should be 19
        #nn.init.xavier_uniform_(self.message.weight)
        nn.init.kaiming_uniform_(self.message.weight, nonlinearity='relu')
        if not discrete:  # This is the general continuous case
            self.activation = nn.Sigmoid()
        elif discrete and end_to_end:  # Discrete and end to end differentiable, so utilise gumbel_softmax
            self.activation = lambda x: F.gumbel_softmax(x, tau=5.0, hard=False)  # Not sure how to set hard variable?
        elif discrete and not end_to_end:  # Discrete and not differentiable
            self.activation = lambda x: F.log_softmax(x, dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        #x = self.dropout(F.relu(self.fc2(x)))
        return self.activation(self.message(x))


class ListenerNet(pl.LightningModule):

    def __init__(self, m_dim=10):
        super(ListenerNet, self).__init__()
        self.fc1 = nn.Linear(1024 + m_dim, 1024)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        #self.fc2 = nn.Linear(1024, 512)
        #nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.logits = nn.Linear(1024, 19)  # this should be 19
        nn.init.kaiming_uniform_(self.logits.weight, nonlinearity='relu')

    def forward(self, x, m=None):
        if m is not None:
            x = torch.cat((x, m), dim=1)  # concat message
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        return F.log_softmax(self.logits(x), dim=1)


class Speaker(pl.LightningModule):

    def __init__(self, m_dim=10, discrete=False, end_to_end=False, dropout=0.0):
        super(Speaker, self).__init__()
        self.CNN = MNISTconv()
        self.FC = SpeakerNet(m_dim, discrete=discrete, end_to_end=end_to_end, dropout=dropout)

    def forward(self, x):
        return self.FC(self.CNN(x))


class Listener(pl.LightningModule):

    def __init__(self, m_dim=10):
        super(Listener, self).__init__()
        self.CNN = MNISTconv()
        self.FC = ListenerNet(m_dim)

    def forward(self, x, m):
        return self.FC(self.CNN(x), m)