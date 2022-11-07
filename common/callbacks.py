import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

class LogPredictionSamplesCallback(Callback):
    def __init__(self, wandb_logger):
        super(LogPredictionSamplesCallback, self).__init__()
        self.my_logger = wandb_logger
        self.data_for_table = []
        self.column_for_m_table = []
        self.test_count = 0

    def on_test_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends.
        A confusion matrix would probably be nice. Got data for this, I think
        So would a message table. Got this!
        It would be super cool to know how the listener classifies randomly sampled messages. Got this!
        Other metrics to include, perplexity was mentioned but I'm not sure what this is of.
        Positive Listening and Positive Signalling stuff would be nice. Should probably be in training loop.
        Should also add some kind of t-sne on the messages so I know what the clusters look like
        """

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        self.test_count += 1
        x1, y1 = batch['speaker']
        x2, y2 = batch['listener']
        y = y1 + y2
        answer, m, m_corrupt = outputs
        data = [self.flatten_list([m_ind, m_c_ind, [y1_i], [y2_i], [o_i], [y_i]])
                for m_ind, m_c_ind, y1_i, y2_i, o_i, y_i
                in list(zip(m.tolist(), m_corrupt.tolist(), y1.tolist(), y2.tolist(), answer.tolist(), y.tolist()))]
        self.data_for_table.extend(data)

        if batch_idx == 0:
            # Option 2: log images and predictions as a W&B Table
            self.column_for_m_table = self.flatten_list([['m_%s'%i for i in range(len(m[0, :]))], ['m_c_%s' %i for i in
                                                          range(len(m_corrupt[0, :]))], ['speaker_image'],
                                                         ['listener_image'], ['prediction'], ['ground truth']])
            columns = ['speaker_image', 'listener_image', 'prediction', 'ground truth' ]
            data = [[wandb.Image(x_1i), wandb.Image(x_2i), y_pred, y_i] for x_1i, x_2i, y_i, y_pred in
                    list(zip(x1, x2, y, answer))]
            self.my_logger.log_table(
                key='Test_Table',
                columns=columns,
                data=data)

    @staticmethod
    def flatten_list(t):
        return [item for sublist in t for item in sublist]

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.data_for_table = []
        self.test_count = 0

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.data_for_table = np.array(self.data_for_table)
        data_embed = TSNE(n_components=2,
                          init='random').fit_transform(self.data_for_table[:,
                                                       :int(self.column_for_m_table.index('speaker_image')/2)])

        self.data_for_table = np.hstack((self.data_for_table, data_embed))
        self.column_for_m_table.extend(['TSNE_x', 'TSNE_y'])
        self.my_logger.log_table(
            key='Message data',
            columns=self.column_for_m_table,
            data=self.data_for_table)


class MessageDimensionCallbacks(LogPredictionSamplesCallback):
    """
    I'm interested in digit/message purity which I'm observing through entropy
    """

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.validation_data = []

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x1, y1 = batch['speaker']
        x2, y2 = batch['listener']
        y = y1 + y2
        answer, m, _ = outputs
        val_data = [[m_i, y_i] for (m_i, y_i) in zip(m.tolist(), y1.tolist())]
        self.validation_data.extend(val_data)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        I want two things from this, the purity of the symbols themselves.
        Secondly, I want the purity of the digits. Are they represented by the same number
        It would be quite nice to have a histogram of message frequency.
        I'd probably also just like mutual information here too.
        :param trainer:
        :param pl_module:
        :return:
        """
        self.validation_data = np.array(self.validation_data)
        df = pd.DataFrame({'m': self.validation_data[:, 0], 'd': self.validation_data[:, 1]})
        # H(M)
        p_m = df['m'].value_counts(normalize=True).to_numpy()
        H_M = (-p_m * np.log(p_m)).sum()
        # H(D)
        p_d = df['d'].value_counts(normalize=True).to_numpy()
        H_D = (-p_d * np.log(p_d)).sum()
        # H(M | D)
        H_M_given_D = 0
        h_m_given_d_is_d = [0] * 10
        j = 0
        for i in range(10):
            p_m_given_d_is_d = df.loc[df['d'] == i].value_counts(normalize=True).to_numpy()
            if not p_m_given_d_is_d.size == 0:
                h_m_given_d_is_d[i] = (-p_m_given_d_is_d * np.log(p_m_given_d_is_d)).sum()
                H_M_given_D += p_d[j] * h_m_given_d_is_d[i]
                j += 1
        # H(D | M)
        H_D_given_M = 0
        h_d_given_m_is_m = [0] * pl_module.hparams.m_dim
        j = 0
        for i in range(pl_module.hparams.m_dim):
            p_d_given_m_is_m = df.loc[df['m'] == i].value_counts(normalize=True).to_numpy()
            if not p_d_given_m_is_m.size == 0:
                h_d_given_m_is_m[i] = (-p_d_given_m_is_m * np.log(p_d_given_m_is_m)).sum()
                H_D_given_M += p_m[j] * h_d_given_m_is_m[i]
                j += 1
        I_M_D = H_M - H_M_given_D  # Mutual information I(M,D)
        I_D_M = H_D - H_D_given_M  # Mutual Information I(D,M)
        cardinality = df['m'].nunique()
        for_logging = {'d%i' % num: i for num, i in enumerate(h_m_given_d_is_d)}
        for_logging.update({'m%i' % num: i for num, i in enumerate(h_d_given_m_is_m)})
        for_logging.update({'cardinality': cardinality, 'I(M,D)': I_M_D,
                            'I_D_M': I_D_M})
        message_counts = {str(i): 0 for i in range(pl_module.hparams.m_dim)}
        message_counts.update({str(i): j for (i, j) in zip(df['m'].unique(), df['m'].value_counts().to_numpy())})
        for_logging.update(message_counts)
        pl_module.log('val_extra_metrics', for_logging)












