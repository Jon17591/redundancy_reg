# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torchvision.datasets import KMNIST
from torchvision import transforms


class RandomMessage(Dataset):
    def __init__(self, data_len, m_dim):
        super(RandomMessage, self).__init__()
        self.values = torch.rand(data_len, m_dim)

    def __len__(self):
        return len(self.values)  # number of samples in the dataset

    def __getitem__(self, index):
        return self.values[index]

class TreasureHuntEnv(Dataset):
    def __init__(self, data_len, m_dim):
        super(TreasureHuntEnv, self).__init__()
        self.env = TreasureHunt()

    def __len__(self):
        return 1  # number of samples in the dataset

    def __getitem__(self, index):
        return env.reset()





class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, m_dim=10, batch_size=32, normalise_data=True, data_dir: str = "./", mnist_variant='mnist'):
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.normalise_data:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def mnist_variant(self, train=True):
        if self.hparams.mnist_variant == 'mnist':
            return MNIST(self.hparams.data_dir, train=train, transform=self.transform)
        elif self.hparams.mnist_variant == 'fashion_mnist':
            return FashionMNIST(self.hparams.data_dir, train=train, transform=self.transform)
        elif self.hparams.mnist_variant == 'kmnist':
            return KMNIST(self.hparams.data_dir, train=train, transform=self.transform)
        else:
            return MNIST(self.hparams.data_dir, train=train, transform=self.transform)

    def prepare_data(self):
        # download
        MNIST(self.hparams.data_dir, train=False, download=True)
        KMNIST(self.hparams.data_dir, train=False, download=True)
        FashionMNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders, note insertion of split for EMNIST
        if stage == "fit" or stage is None:
            mnist_speaker_full = self.mnist_variant()
            mnist_listener_full = self.mnist_variant()
            train_split = int(len(mnist_speaker_full) * ((60000 - 5000)/60000))
            val_split = int(len(mnist_speaker_full) - train_split)
            self.mnist_speaker_train, self.mnist_speaker_val = random_split(mnist_speaker_full, [train_split, val_split])
            self.mnist_listener_train, self.mnist_listener_val = random_split(mnist_listener_full, [train_split, val_split])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_speaker_test = self.mnist_variant(train=False)
            self.mnist_listener_test = self.mnist_variant(train=False)
            self.random_message = RandomMessage(self.mnist_listener_test.__len__(), self.hparams.m_dim)

        if stage == "predict" or stage is None:
            self.mnist_speaker_predict = self.mnist_variant(train=False)
            self.mnist_listener_predict = self.mnist_variant(train=False)

    def train_dataloader(self):
        return CombinedLoader({'speaker': DataLoader(self.mnist_speaker_train,
                                                     shuffle=True,
                                                     batch_size=self.hparams.batch_size),
                               'listener': DataLoader(self.mnist_listener_train,
                                                      shuffle=True,
                                                      batch_size=self.hparams.batch_size)})

    def val_dataloader(self):
        return CombinedLoader({'speaker': DataLoader(self.mnist_speaker_val,
                                                     shuffle=True,
                                                     batch_size=self.hparams.batch_size),
                               'listener': DataLoader(self.mnist_listener_val,
                                                      shuffle=True,
                                                      batch_size=self.hparams.batch_size)})

    def test_dataloader(self):
        """
        return CombinedLoader({'speaker': DataLoader(self.mnist_speaker_test,
                                                     batch_size=self.hparams.batch_size),
                               'listener': DataLoader(self.mnist_listener_test,
                                                      batch_size=self.hparams.batch_size),
                               'rnd_m': DataLoader(self.random_message,
                                                   batch_size=self.hparams.batch_size)})
        """
        #speaker_test = Subset(self.mnist_speaker_train, torch.arange(320))
        #listener_test = Subset(self.mnist_listener_train, torch.arange(320))
        return CombinedLoader({'speaker': DataLoader(self.mnist_speaker_test,
                                                     shuffle=True,
                                                     batch_size=self.hparams.batch_size),
                               'listener': DataLoader(self.mnist_listener_test,
                                                      shuffle=True,
                                                      batch_size=self.hparams.batch_size),
                               'rnd_m': DataLoader(self.random_message,
                                                   shuffle=True,
                                                   batch_size=self.hparams.batch_size)})

    def predict_dataloader(self):
        return CombinedLoader({'speaker': DataLoader(self.mnist_speaker_predict,
                                                     shuffle=True,
                                                     batch_size=self.hparams.batch_size),
                               'listener': DataLoader(self.mnist_listener_predict,
                                                      shuffle=True,
                                                      batch_size=self.hparams.batch_size)})


class MNISTDataModule_EVAL(MNISTDataModule):
    """
    This abomination is so that I can evaluate performance on the test data every iteration.
    It is not pretty, but it allows for the functionality that I need.
    I don't actually think this is a good idea, probably TBD going to do it on the validation dataset instead.
    """

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_speaker_full = MNIST(self.hparams.data_dir, train=True, transform=self.transform)
            mnist_listener_full = MNIST(self.hparams.data_dir, train=True, transform=self.transform)
            self.mnist_speaker_train, _ = random_split(mnist_speaker_full, [55000, 5000])
            self.mnist_listener_train, _ = random_split(mnist_listener_full, [55000, 5000])
            self.mnist_speaker_val = MNIST(self.hparams.data_dir, train=False, transform=self.transform)
            self.mnist_listener_val = MNIST(self.hparams.data_dir, train=False, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_speaker_test = MNIST(self.hparams.data_dir, train=False, transform=self.transform)
            self.mnist_listener_test = MNIST(self.hparams.data_dir, train=False, transform=self.transform)
            self.random_message = RandomMessage(self.mnist_listener_test.__len__(), self.hparams.m_dim)

        if stage == "predict" or stage is None:
            self.mnist_speaker_predict = MNIST(self.hparams.data_dir, train=False, transform=self.transform)
            self.mnist_listener_predict = MNIST(self.hparams.data_dir, train=False, transform=self.transform)
