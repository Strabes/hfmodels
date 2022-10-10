from transformers import (
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaTokenizerFast)
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

N_RECORDS = 1000

dataset = load_dataset("imdb")

tokenizer = RobertaTokenizerFast.from_pretrained(
    "./models/cnn_dm_bpe", max_len=512,
    pad_token = "<pad>",
    bos_token = "<s>",
    eos_token = "</s>",
    unk_token = "<unk>",
    mask_token = "<mask>")

model = RobertaForSequenceClassification.from_pretrained(
    "models/cnn_dm_roberta",
    num_labels=len(set(dataset["train"]["label"]))
)

def tokenization(batched_text):
    return tokenizer(
        batched_text["text"],
        padding='max_length',
        truncation=True,
        max_length=512)

train_dataset = (dataset["train"]
    .select(range(N_RECORDS))
    .map(tokenization, batched=True, batch_size = 16))

eval_dataset = (dataset["test"]
    .select(range(N_RECORDS))
    .map(tokenization, batched=True, batch_size = 16))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1.tolist(),
        'precision': precision.tolist(),
        'recall' : recall.tolist()
    }

training_args = TrainingArguments(
    output_dir = 'models/finetuned_roberta',
    num_train_epochs = 5,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 8,
    per_device_eval_batch_size = 16,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    disable_tqdm = False,
    load_best_model_at_end=True,
    warmup_steps=200,
    weight_decay=0.01,
    logging_steps=4,
    logging_dir='models/finetuned_roberta',
    dataloader_num_workers=0,
    run_name='finetune_roberta'
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset)

trainer.train()

model.eval()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cuda")

tokenized_example = {
    k: v.to(device) for k,v in tokenizer(
    "This movie sucks.",
    return_tensors='pt').items()}

model(**tokenized_example)