import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
import pytorch_lightning as pl



class WeightedVoting(nn.Module):
    def __init__(self, num_model=5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_model, requires_grad=True).float())
        self.weight[1] = 2
        # self.register_parameter("weight", weight)
    
    def forward(self, x):
        """
        x: [batch_size, num_models, num_choices]
        output: [batch_size, num_choices]
        """
        t_weight = self.weight.unsqueeze(0).unsqueeze(2)
        x = x * t_weight
        return x
    

class WeightTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = WeightedVoting(config)
        self.loss_fct = F.cross_entropy()

    def configure_optimizers(self):
        return AdamW(self, lr=self.config["lr"])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fct(y_hat, y)

        return {"loss": loss}


inputs = torch.randint(1,5, size=(2,5,4)).float()
a = WeightedVoting(5)

a(inputs)

import IPython; IPython.embed(); exit(1)