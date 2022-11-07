from common.data_stuff import MNISTDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from agents.end_to_end_agents import Communicators
from common.callbacks import LogPredictionSamplesCallback
import torch
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--lr", type=float, default=0.00003)
parser.add_argument("--m_dim", type=int, default=10)
parser.add_argument("--noise_variance", type=float, default=0.0)
parser.add_argument("--method", type=str, default="supervised_learning")
parser.add_argument("--entropy_bonus", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--iteration", type=int, default=0)  # more runs with wandb sweep easily.
parser.add_argument("--gradient_clipping", type=float, default=1.0)
parser.add_argument("--normalise_data", type=bool, default=False)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--discrete", type=bool, default=True)
parser.add_argument("--end_to_end", type=bool, default=True)  # Todo: Implement this feature

hparams = parser.parse_args()
hparams = vars(hparams)

wandb_logger = WandbLogger(project='MNIST', log_model=False, name='SL_gumbel_softmax')
mnist_dm = MNISTDataModule(m_dim=hparams['m_dim'], batch_size=hparams['batch_size'],
                           normalise_data=hparams['normalise_data'])
trainer = pl.Trainer(gpus=[2], max_epochs=hparams['epochs'], logger=wandb_logger, log_every_n_steps=100,
                     callbacks=[LogPredictionSamplesCallback(wandb_logger), ],
                     gradient_clip_val=hparams['gradient_clipping'])
model = Communicators(m_dim=hparams['m_dim'], lr=hparams['lr'],
                      noise_variance=hparams['noise_variance'], method=hparams['method'],
                      entropy_bonus=hparams['entropy_bonus'], gumbel_softmax=hparams['gumbel_softmax'],
                      end_to_end=hparams['end_to_end'])
wandb_logger.watch(model)
trainer.fit(model, datamodule=mnist_dm)
trainer.test(model, datamodule=mnist_dm)
#trainer.validate(model, datamodule=mnist_dm)


