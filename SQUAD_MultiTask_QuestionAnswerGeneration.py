from typing import Tuple
from pandas import DataFrame, read_csv
import Constants as c
from QuestionAnswerDataModule import QuestionAnswerDataModule
from QuestionAnswerModelGeneration import QuestionAnswerModel
from transformers import T5TokenizerFast as T5Tokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Todo
# 1. Read Datasets : SQUAD Train, SQUAD Validation [x]
# 2. Prepare Datasets to train/val using Pytorch Lightning [x]
# 3. Prepare the model : Input for encoder and decoder [x]
# 4. Test the code [Later]


def preprocessDataset(dataset: DataFrame) -> None:
    df: DataFrame = dataset.copy()
    df.dropna(inplace=True)
    df.rename(columns={"context_para": c.CONTEXT_PARAGRAPH_QG},
              inplace=True)
    df.drop(columns=["context_sent",
                     "answer_start",
                     "answer_end"], inplace=True)
    return df


def splitDataset(squadTrain: DataFrame) -> Tuple[DataFrame, DataFrame]:

    valDf: DataFrame = squadTrain[:11877]
    trainDf: DataFrame = squadTrain[11877:]

    return trainDf, valDf


squadTrain: DataFrame = read_csv("train_df.csv")
squadTest: DataFrame = read_csv("dev_df.csv")

prepTrain: DataFrame = preprocessDataset(squadTrain)
prepTest: DataFrame = preprocessDataset(squadTest)

prepTrain, prepVal = splitDataset(prepTrain)

tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(c.MODEL_NAME)
tokenizer.add_tokens(c.SEP_TOKEN)
QAD: QuestionAnswerDataModule = QuestionAnswerDataModule(prepTrain,
                                                         prepVal,
                                                         prepTest,
                                                         tokenizer,
                                                         c.SOURCE_MAX_TOKEN_QG,
                                                         c.TARGET_MAX_TOKEN_QG,
                                                         )

QAM: QuestionAnswerModel = QuestionAnswerModel()
checkpointCallback: ModelCheckpoint = ModelCheckpoint(
    dirpath="QAP",
    filename="best-checkpoint",
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    verbose=True)

trainer = pl.Trainer(
    checkpoint_callback=checkpointCallback,
    max_epochs=c.EPOCHS,
    gpus=1,
    progress_bar_refresh_rate=30)
