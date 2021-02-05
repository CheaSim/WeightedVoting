import torch
import torch.nn as nn
from torch.utils.data import  TensorDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import random

import argparse
parser = argparse.ArgumentParser()




class WeightedVoting(nn.Module):
    def __init__(self, num_model=5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_model, requires_grad=True).float())
        # self.l1 = nn.Linear(123,321)
        # nn.LayerNorm
        # self.weight[1] = 2
        # self.register_parameter("weight", self.weight)
    
    def forward(self, x):
        """
        x: [batch_size, num_models, num_choices]
        output: [batch_size, num_choices]
        """
        t_weight = self.weight.unsqueeze(0).unsqueeze(2)
        x = x * t_weight
        x = torch.mean(x, dim=1)
        return x
    
class TrainWeightedVoting(pl.LightningModule):
    def __init__(self, config, train_set, test_set):
        super().__init__()

        assert len(train_set.shape) == 3, "input dims must be [num_samples, num_models, num_choies]"
        assert len(test_set.shape) == 3, "input dims must be [num_samples, num_models, num_choies]"

        num_models = train_set.shape[1]
        self.config = config
        self.hparams.lr = config["lr"]
        self.hparams.batch_size = 12
        self.train_set, self.val_set = self._split_train(train_set)
        self.test_set = test_set
        # self.save_hyperparameters()
        self.model = WeightedVoting(num_models)
        self.loss_fct = nn.CrossEntropyLoss()
        # import IPython; IPython.embed(); exit(1)

    def forward(self, x):
        # import IPython; IPython.embed(); exit(1)
        return self.model(x)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fct(y_hat, y)

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # import IPython; IPython.embed(); exit(1)
        y_hat = self.model(x)
        loss = self.loss_fct(y_hat, y)

        return {"eval_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fct(y_hat, y)

        return {"eval_loss": loss}

    def train_dataloader(self):
        dataloader = DataLoader(self.train_ds, shuffle=True, batch_size=self.hparams.batch_size)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_ds, shuffle=False, batch_size=self.hparams.batch_size)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.test_ds, shuffle=False, batch_size=self.hparams.batch_size)
        return dataloader

    def _split_train(self, train):
        total_num_sample = train.shape[0]
        index = list(range(total_num_sample))
        random.shuffle(index)
        train = train[index]

        return train[:(total_num_sample//10)*9], train[(total_num_sample//10)*9:]

    def prepare_data(self) -> None:
        # import IPython; IPython.embed(); exit(1)
        # x, y = [num_samples, num_models, num_choices], [num_samples, 1]
        self.train_ds = TensorDataset(self.train_set[:,:,:-1], torch.mean(self.train_set,dim=[1,2]).long())
        self.val_ds = TensorDataset(self.val_set[:,:,:-1], torch.mean(self.val_set,dim=[1,2]).long())
        self.test_ds = TensorDataset(self.test_set)


    # for adjusting the batch size and learning rate
    @property
    def batch_size(self): return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, batch_size): self.hparams.batch_size = batch_size

    @property
    def lr(self): return self.hparams.lr

    @lr.setter
    def lr(self, lr): self.hparams.lr = lr

# a = WeightedVoting(5)
# inputs = torch.randint(1,5, size=(2,5,4)).float()
# a(inputs)
# print(torch.__version__)
if __name__ == "__main__":
    train_set = torch.ones([1000,3,6])
    test_set = torch.rand([100,3,5])
    model = TrainWeightedVoting(dict(lr=1e-5),train_set, test_set)

    trainer_args = dict(
        gpus = 1,
        limit_train_batches=100,
        # epochs= 10,        
        check_val_every_n_epoch=1,
    )
    
    trainer = Trainer(**trainer_args)

    trainer.fit(model)
    trainer.test()

    # import IPython; IPython.embed(); exit(1)