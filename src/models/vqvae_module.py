from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.vqvae import VQVAE

class VQVAEModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # for averaging loss across batches
        self.train_mse = MeanMetric()
        self.train_emb_loss = MeanMetric()
        self.train_ppl = MeanMetric()
        # self.val_loss = MeanMetric()
        # self.test_mse = MeanMetric()
        self.test_emb_loss = MeanMetric()
        self.test_ppl = MeanMetric()

        # for tracking best so far
        self.test_mse = MinMetric()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.test_emb_loss = MeanMetric()
        self.test_ppl = MeanMetric()

    
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x = batch
        embedding_loss, x_hat, ppl = self.forward(x.unsqueeze(-1))
        mse_loss = self.criterion(x_hat.view(-1), x.view(-1))
        # preds = torch.argmax(logits, dim=1)
        return embedding_loss, mse_loss, ppl
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        embedding_loss, mse_loss, ppl = self.model_step(batch)
        # update and log metrics
        self.train_mse(mse_loss)
        self.train_emb_loss(embedding_loss)
        self.train_ppl(ppl)
        # self.train_acc(preds, targets)
        self.log("train/mse", self.train_mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/emb_loss", self.train_emb_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ppl", self.train_ppl, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return mse_loss + embedding_loss
    
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        embedding_loss, mse_loss, ppl = self.model_step(batch)

        # update and log metrics
        self.test_mse(mse_loss)
        self.test_emb_loss(embedding_loss)
        self.train_ppl(ppl)
        # self.test_acc(preds, targets)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/emb_loss", self.test_emb_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/ppl", self.test_ppl, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "test/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
if __name__ == "__main__":
    _ = VQVAEModule(None, None, None, None)

    

