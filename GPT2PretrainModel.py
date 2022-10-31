from transformers import GPT2LMHeadModel
from transformers.optimization import AdamW
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

class GPT2PretrainModel(LightningModule):

    def __init__(self, config, lr: float,):
        super().__init__()
        self.model = GPT2LMHeadModel(config)
        self.lr = lr

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def training_step(self, inputs, batch_idx):
        # outputs: CausalLMOutputWithCrossAttentions
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        self.log("loss", outputs.loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return outputs.loss

    def validation_step(self, inputs, batch_idx):
        # outputs: CausalLMOutputWithCrossAttentions
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        self.log("val_loss", outputs.loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return outputs.loss