import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, AdamW
import Constants as c
import torch
from typing import Dict


class QuestionAnswerModel(pl.LightningModule):
    def __init__(this):
        super().__init__()
        this.model = T5ForConditionalGeneration.from_pretrained(c.MODEL_NAME,
                                                                return_dict=True)
        this.model.resize_token_embeddings(c.NEW_TOKENIZER_SIZE)

    def forward(this, input_ids, attention_mask, labels=None):
        """
            This is from dataset
                Refer to QuestionAnswerDataModule
        """

        output: torch.Tensor = this.model(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          labels=labels)
        return output.loss, output.logits

    def training_step(this, batch: Dict, batch_idx):
        inputIds: torch.Tensor = batch["input_ids"]
        attentionMask: torch.Tensor = batch["attention_mask"]
        labels: torch.Tensor = batch["labels"]
        loss, logits = this(inputIds, attentionMask, labels)
        this.log("train_Loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(this, batch: Dict, batch_idx):
        inputIds: torch.Tensor = batch["input_ids"]
        attentionMask: torch.Tensor = batch["attention_mask"]
        labels: torch.Tensor = batch["labels"]
        loss, logits = this(inputIds,
                            attentionMask,
                            labels)
        this.log("val_Loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(this, batch: Dict, batch_idx):
        inputIds: torch.Tensor = batch["input_ids"]
        attentionMask: torch.Tensor = batch["attention_mask"]
        labels: torch.Tensor = batch["labels"]
        loss, logits = this(inputIds,
                            attentionMask,
                            labels)
        this.log("test_Loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(this):
        return AdamW(this.parameters(), lr=c.LEARNING_RATE)
