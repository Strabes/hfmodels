from pathlib import Path
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torch
torch.cuda.is_available()
from transformers import (
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline)

VOCAB_SIZE = 3000
N_RECORDS = 20_000

dataset = load_dataset("cnn_dailymail", version="3.0.0")

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train_from_iterator(
    dataset["train"]["article"][0:N_RECORDS],
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("models/cnn_dm_bpe")


tokenizer = ByteLevelBPETokenizer(
    "./models/cnn_dm_bpe/vocab.json",
    "./models/cnn_dm_bpe/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

tokenizer.encode("This story is about politics.")

tokenizer.encode("This story is about politics.").tokens


config = RobertaConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=8,
    max_position_embeddings=514,
    num_attention_heads=2,
    num_hidden_layers=2,
    intermediate_size=8,
    type_vocab_size=1,
    pad_token_id=tokenizer.token_to_id("<pad>"),
    bos_token_id=tokenizer.token_to_id("<s>"),
    eos_token_id=tokenizer.token_to_id("</s>")
)

tokenizer = RobertaTokenizerFast.from_pretrained(
    "./models/cnn_dm_bpe", max_len=512,
    pad_token = "<pad>",
    bos_token = "<s>",
    eos_token = "</s>",
    unk_token = "<unk>",
    mask_token = "<mask>")

model = RobertaForMaskedLM(config=config)

model.num_parameters()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./model/cnn_dm_roberta",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_gpu_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True
)

def tokenization(batched_text):
    return tokenizer(
        batched_text["article"],
        padding='max_length',
        truncation=True,
        max_length=512)

train_dataset = (dataset["train"]
    .select(range(N_RECORDS))
    .map(tokenization, batched=True, batch_size = 16))

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

trainer.save_model("./models/cnn_dm_roberta")

fill_mask = pipeline(
    "fill-mask",
    model="./models/cnn_dm_roberta",
    tokenizer=RobertaTokenizerFast.from_pretrained("./models/cnn_dm_bpe", max_len=512)
)

fill_mask("The sun <mask>.")

fill_mask("This is some <mask> text.")