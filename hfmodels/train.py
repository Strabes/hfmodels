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
import json

from tokenizers.processors import BertProcessing

from tqdm.auto import tqdm

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordPiece

dataset = load_dataset("cnn_dailymail", version="3.0.0")
#paths = [str(x) for x in Path("./eo_data/").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
tok = Tokenizer(SentencePieceUnigramTokenizer())

tokenizer.train_from_iterator(
    dataset["train"]["article"][0:1000],
    vocab_size=2000,
    unk_token="<unk>",
    special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("models/cnn_dm_sent_piece")

with open("models/cnn_dm_sent_piece/unigram.json","r",encoding="utf8") as f:
    sp = json.load(f)

tok = Unigram(vocab=[(i[0],i[1]) for i in sp["vocab"]])
tok.tokenize("Here is some text.")

tokenizer = Tokenizer.from_file("models/cnn_dm_sent_piece/unigram.json")
encoded = tokenizer.encode("Here is some text.")
encoded.tokens
# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])



length = 100000
dataset_name = 'transformersbook/codeparrot-train'
dataset = load_dataset(dataset_name, split="train", streaming=True)
iter_dataset = iter(dataset)

def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, length, batch_size)):
        yield [next(iter_dataset)['content'] for _ in range(batch_size)]

new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), 
                                                  vocab_size=12500,
                                                  initial_alphabet=base_vocab)