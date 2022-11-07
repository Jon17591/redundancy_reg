from common.data_stuff import MNISTDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from agents.independent_agents import Communicators
from common.callbacks import LogPredictionSamplesCallback, MessageDimensionCallbacks
import torch
from argparse import ArgumentParser
import os

parser = ArgumentParser()

parser.add_argument("--m_dim", type=int, default=20)
parser.add_argument("--noise_variance", type=float, default=0.0)
parser.add_argument("--method", type=str, default="reinforcement_learning")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=5)  # More runs with wandb sweep easily.
parser.add_argument("--normalise_data", type=bool, default=False)
parser.add_argument("--epochs", type=int, default=500) #100
parser.add_argument("--discrete", type=bool, default=True)
parser.add_argument("--speaker_lr", type=float, default=0.0003)
parser.add_argument("--listener_lr", type=float, default=0.0003)
parser.add_argument("--ps_weight", type=float, default=0.1)
parser.add_argument("--pl_weight", type=float, default=0.01)
parser.add_argument("--speaker_ent_target", type=float, default=1.0)
parser.add_argument("--speaker_ent_bonus", type=float, default=0.0)
parser.add_argument("--speaker_lambda", type=float, default=0.3)
parser.add_argument("--listener_ent_bonus", type=float, default=0.03)
parser.add_argument("--ce_loss", type=float, default=0.001)
parser.add_argument("--gradient_clipping", type=float, default=1.0)
parser.add_argument("--ps_entropy", type=str, default='linear')
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--ps_ent_switch", type=int, default=300)
parser.add_argument("--dataset", type=str, default='kmnist')


hparams = parser.parse_args()
hparams = vars(hparams)
pl.seed_everything(hparams['seed'])
torch.backends.cudnn.determinstic = True
#torch.backends.cudnn.benchmark = False
wandb_logger = WandbLogger(project='Message_Dimension_Investigation', log_model=False,
                           name='m_dim:%s_iter:%s'%(hparams['m_dim'], hparams['seed']))
mnist_dm = MNISTDataModule(m_dim=hparams['m_dim'], batch_size=hparams['batch_size'],
                           normalise_data=hparams['normalise_data'], mnist_variant=hparams['dataset'])

# get gpu
my_directory = int(os.getcwd().split('_')[-1])
if my_directory == 623:
    my_gpu = 2
else:
    my_gpu = my_directory

trainer = pl.Trainer(gpus=[my_gpu], max_epochs=hparams['epochs'], logger=wandb_logger, log_every_n_steps=100,
                     callbacks=[MessageDimensionCallbacks(wandb_logger), ],)
model = Communicators(m_dim=hparams['m_dim'], noise_variance=hparams['noise_variance'], method=hparams['method'],
                      discrete=hparams['discrete'], speaker_lr=hparams['speaker_lr'], listener_lr=hparams['listener_lr']
                      , ps_weight=hparams['ps_weight'], pl_weight=hparams['pl_weight'], ce_loss=hparams['ce_loss'],
                      speaker_ent_target=hparams['speaker_ent_target'], speaker_ent_bonus=hparams['speaker_ent_bonus'],
                      speaker_lambda=hparams['speaker_lambda'], listener_ent_bonus=hparams['listener_ent_bonus'],
                      gradient_clipping=hparams['gradient_clipping'], ps_entropy=hparams['ps_entropy'],
                      dropout=hparams['dropout'], weight_decay=hparams['weight_decay'],
                      ps_ent_switch=hparams['ps_ent_switch'])
wandb_logger.watch(model)
trainer.fit(model, datamodule=mnist_dm)
trainer.test(model, datamodule=mnist_dm)
trainer.save_checkpoint("models/my_communicators_%s_%s.ckpt"%(hparams['m_dim'], hparams['seed']))
#trainer.validate(model, datamodule=mnist_dm)


