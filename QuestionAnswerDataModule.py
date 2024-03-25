import pytorch_lightning as pl
from numpy.random import rand
from pandas import DataFrame, Series
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from transformers import T5TokenizerFast as T5Tokenizer, BatchEncoding
import Constants as c


class QuestionAnswerDataset(Dataset):
    def __init__(this,
                 data: DataFrame,
                 tokenizer: T5Tokenizer,
                 sourceMaxTokenLength: int, 
                 targetMaxTokenLength: int,
                 ):
        this.tokenizer: T5Tokenizer = tokenizer
        this.data: DataFrame = data 
        this.sourceMaxTokenLength: int = sourceMaxTokenLength
        this.targetMaxTokenLength: int = targetMaxTokenLength

    def __len__(this):
        return len(this.data)

    def __getitem__(this, index: int):
        """
            Get 1 item from dataset. Including the features and target
                Question - Answer Pairs
                Question need to be tokenizer
                Answer need to be Tokenizer
                There are some Answer that is Not provided by using masking chance

            Return : A Dictionary containing item for each observations.
            The importants are :
                1. InputIds FeaturesEncoding
                2. AttentionMask FeaturesEncoding
                3. Target / labels
            """
        inputRow: Series = this.data.iloc[index]
        answer: str = "[MASK]"

        if rand() > c.MASKING_CHANCE:
            answer = inputRow[c.ANSWER_TEXT_QG]

        textInput: str = "{} {} {}".format(answer,
                                           c.SEP_TOKEN,
                                           inputRow[c.CONTEXT_PARAGRAPH_QG])

        targetInput: str = "{} {} {}".format(inputRow[c.ANSWER_TEXT_QG],
                                             c.SEP_TOKEN,
                                             inputRow[c.QUESTION_QG])

        featuresEncoding: BatchEncoding = this.tokenizer(textInput,
                                                         max_length=this.sourceMaxTokenLength,
                                                         padding="max_length",
                                                         truncation="true",
                                                         add_special_tokens=True,
                                                         return_tensors="pt"
                                                         )

        targetEncoding: BatchEncoding = this.tokenizer(targetInput,
                                                       max_length=this.targetMaxTokenLength,
                                                       padding="max_length",
                                                       truncation=True,
                                                       add_special_tokens=True,
                                                       return_tensors="pt")
        labels = targetEncoding["input_ids"]
        labels[labels == 0] = -100

        itemToReturn: Dict = dict(
            answer_text=inputRow[c.ANSWER_TEXT_QG],
            context=inputRow[c.CONTEXT_PARAGRAPH_QG],
            question=inputRow[c.QUESTION_QG],
            inputIds=featuresEncoding["input_ids"].flatten(),
            attentionMask=featuresEncoding["attention_mask"].flatten(),
            labels=labels.flatten()
        )

        return itemToReturn


class QuestionAnswerDataModule(pl.LightningDataModule):

    def __init__(this,
                 trainDf: DataFrame,
                 valDf: DataFrame,
                 testDf: DataFrame,
                 tokenizer: T5Tokenizer,
                 sourceMaxTokenLength: int,
                 targetMaxTokenLength: int,
                 batchSize: int,
                 ):
        super.__init__()
        this.trainDf: DataFrame = trainDf
        this.valDf: DataFrame = valDf
        this.testDf: DataFrame = testDf
        this.tokenizer: T5Tokenizer = tokenizer
        this.sourceMaxTokenLength: int = sourceMaxTokenLength,
        this.targetMaxTokenLength: int = targetMaxTokenLength
        this.batchSize: int = batchSize

        this.trainDataset: Dataset = None
        this.validDataset: Dataset = None
        this.testDataset: Dataset = None

    def setup(this):
        this.trainDataset = QuestionAnswerDataset(this.trainDf,
                                                  this.tokenizer,
                                                  this.sourceMaxTokenLength,
                                                  this.targetMaxTokenLength,
                                                  )

        this.validDataset = QuestionAnswerDataset(this.valDf,
                                                  this.tokenizer,
                                                  this.sourceMaxTokenLength,
                                                  this.targetMaxTokenLength)

        this.testDataset = QuestionAnswerDataset(this.testDataset,
                                                 this.tokenizer,
                                                 this.sourceMaxTokenLength,
                                                 this.targetMaxTokenLength)

    def train_dataloader(this,) -> DataLoader:
        return DataLoader(this.trainDataset, this.batchSize,
                          shuffle=True, num_workers=2)

    def val_dataloader(this,) -> DataLoader:
        return DataLoader(this.validDataset, this.batchSize)

    def test_dataloader(this,) -> DataLoader:
        return DataLoader(this.testDataset, this.batchSize)
