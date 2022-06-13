from datasets import (
    get_dataset_config_names,
    load_dataset)
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline)
import torch
import time

model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)

domains = get_dataset_config_names('subjqa')

subjqa = load_dataset("subjqa",name="books")

dfs = {split: dset.to_pandas() for split, dset 
    in subjqa.flatten().items()}

for split, df in dfs.items():
    print(f"Number of questions in {split}: {df['id'].nunique()}")

qa_cols = ['title', 'question', 'answers.text',
    'answers.answer_start', 'context']

sample_df = dfs["train"][qa_cols].sample(10, random_state=7)

pipe = pipeline("question-answering",model=model, tokenizer=tokenizer)

start_time = time.time()
train_res = pipe(
    question=subjqa["train"]["question"],
    context=subjqa["train"]["context"])
elapsed_time = time.time() - start_time

for i in range(sample_df.shape[0]):
    print(sample_df.question.tolist()[i])
    print(sample_df.context.tolist()[i])
    print(res[i]['score'])
    print(res[i]['answer'])