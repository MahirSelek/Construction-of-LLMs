import numpy as np
import pandas as pd
import datasets
import torch
import evaluate
import seqeval



from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizerFast
from transformers import TrainerCallback, TrainerControl
from transformers import DataCollatorForTokenClassification

from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset, Dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping




extension = "json"
data_files = {}
data_files["train"] = "/kaggle/input/buster/train.json"  # specify path to train.json
data_files["test"] = "/kaggle/input/buster/test.json"  # specify path to test.json
data_files["validation"] = "/kaggle/input/buster/validation.json"  # specify path to validation.json
raw_dataset = load_dataset(extension, data_files=data_files)


train_dataset = raw_dataset['train']
test_dataset = raw_dataset['test']
validation_dataset = raw_dataset['validation']


id2label = {0:'O',
            1:'B-Generic_Info.ANNUAL_REVENUES',
            2:'I-Generic_Info.ANNUAL_REVENUES',
            3:'B-Advisors.LEGAL_CONSULTING_COMPANY',
            4:'I-Advisors.LEGAL_CONSULTING_COMPANY',
            5:'B-Advisors.GENERIC_CONSULTING_COMPANY',
            6:'I-Advisors.GENERIC_CONSULTING_COMPANY',
            7:'B-Advisors.CONSULTANT', 
            8:'I-Advisors.CONSULTANT',            
            9:'B-Parties.BUYING_COMPANY',
            10:'I-Parties.BUYING_COMPANY',
            11:'B-Parties.ACQUIRED_COMPANY',
            12:'I-Parties.ACQUIRED_COMPANY',
            13:'B-Parties.SELLING_COMPANY',
            14:'I-Parties.SELLING_COMPANY'
           }


label2id = {'O' : 0,
            'B-Generic_Info.ANNUAL_REVENUES' : 1,
            'I-Generic_Info.ANNUAL_REVENUES' : 2,
            'B-Advisors.LEGAL_CONSULTING_COMPANY' : 3,
            'I-Advisors.LEGAL_CONSULTING_COMPANY' : 4,
            'B-Advisors.GENERIC_CONSULTING_COMPANY' : 5,
            'I-Advisors.GENERIC_CONSULTING_COMPANY' : 6,
            'B-Advisors.CONSULTANT' : 7, 
            'I-Advisors.CONSULTANT' : 8,            
            'B-Parties.BUYING_COMPANY' : 9,
            'I-Parties.BUYING_COMPANY' : 10,
            'B-Parties.ACQUIRED_COMPANY' : 11,
            'I-Parties.ACQUIRED_COMPANY' : 12,
            'B-Parties.SELLING_COMPANY' : 13,
            'I-Parties.SELLING_COMPANY' : 14
           }



label_list = ['O',
            'B-Generic_Info.ANNUAL_REVENUES',
            'I-Generic_Info.ANNUAL_REVENUES',
            'B-Advisors.LEGAL_CONSULTING_COMPANY',
            'I-Advisors.LEGAL_CONSULTING_COMPANY',
            'B-Advisors.GENERIC_CONSULTING_COMPANY',
            'I-Advisors.GENERIC_CONSULTING_COMPANY',
            'B-Advisors.CONSULTANT', 
            'I-Advisors.CONSULTANT',            
            'B-Parties.BUYING_COMPANY',
            'I-Parties.BUYING_COMPANY',
            'B-Parties.ACQUIRED_COMPANY',
            'I-Parties.ACQUIRED_COMPANY',
            'B-Parties.SELLING_COMPANY' , 
            'I-Parties.SELLING_COMPANY']



train_labels = train_dataset['labels']
test_labels = test_dataset['labels']
validation_labels = validation_dataset['labels']

train_tags = [[label2id[x] for x in tag] for tag in train_labels]
val_tags = [[label2id[x] for x in tag] for tag in validation_labels]
test_tags = [[label2id[x] for x in tag] for tag in test_labels]

train = pd.DataFrame({'tokens': train_dataset['tokens'], 'labels': train_tags})
val = pd.DataFrame({'tokens':  validation_dataset['tokens'], 'labels': val_tags})
test = pd.DataFrame({'tokens':  test_dataset['tokens'], 'labels': test_tags})

train_dataset = Dataset.from_pandas(train)
validation_dataset = Dataset.from_pandas(val)
test_dataset = Dataset.from_pandas(test)



# Load the pre-trained model and tokenizer

model_checkpoint = "/kaggle/input/checkpoint-12000" #directory of my checkpoint 
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base",add_prefix_space=True, model_max_length = 256)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=15, id2label=id2label, label2id=label2id) # assuming I have 15 NER tags


#Chunking
def get_chunk_truncation_size(doc_txt, tokenizer, max_seq_length=256):
    """
    Given a sequence of tokens (document), pre-computes the appropriate chunk size so that
    it avoids truncation while not wasting computations with too much padding.
    :param doc_txt:
    :param tokenizer:
    :param max_seq_length:
    :return: The position where to truncate doc_txt in a chunk. An integer.
    """

    tokenized_inputs = tokenizer(
            [doc_txt],
            padding=False,
            truncation=False,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True
        )

    if len(tokenized_inputs["input_ids"][0]) < max_seq_length:
        return len(doc_txt)

    trunc_size = 64
    chunk_size = 0
    min_size_increment = 32
    while chunk_size < max_seq_length:
        trunc_size += min_size_increment
        tokenized_inputs = tokenizer(
            [doc_txt[:trunc_size]],
            padding=False,
            truncation=False,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True
        )
        if chunk_size == 0 and len(tokenized_inputs["input_ids"][0]) >= max_seq_length:
            print("WARNING MAX SEQ LENGTH EXCEED AT FIRST STEP=============")

        chunk_size = len(tokenized_inputs["input_ids"][0])

    trunc_size = trunc_size - min_size_increment

    return trunc_size

def chunking(examples, tokenizer, seq_max_length, text_column_name=None, label_column_name=None):
    # split of wide pages in 256 word level chunks
    text_column_name = text_column_name if text_column_name else "tokens"
    label_column_name = label_column_name if label_column_name else "labels"  # FIXME you must remove labels processing from this function
    # FIXME since you shall have already removed

    texts = examples[text_column_name]
    labels = examples[label_column_name]
    doc_ids = examples["document_id"]
    chunked_texts, chunked_labels, chunked_doc_ids, chunk_ids = [], [], [], []
    for i in range(len(texts)):
        doc_text = texts[i]
        doc_labels = labels[i]
        assert len(doc_text) == len(doc_labels), f"Tokens/labels mismatch: {len(doc_text)} and {len(doc_labels)}."

        residual_text, residual_labels = doc_text, doc_labels

        doc_id = doc_ids[i]
        chunk_id = 0
        while len(residual_text) > 0:
            if chunk_id > 0:
                print(f"Creating more than one chunk ({chunk_id}) for {doc_id}")

            chunk_ids.append(chunk_id)
            chunk_len = get_chunk_truncation_size(
                residual_text,
                tokenizer,
                seq_max_length
            )

            chunked_texts.append(residual_text[:chunk_len])
            chunked_labels.append(residual_labels[:chunk_len])
            chunked_doc_ids.append(doc_id)

            residual_text = residual_text[chunk_len:]  # removes first doc_truncation_size tokens
            residual_labels = residual_labels[chunk_len:]  # removes first doc_truncation_size labels
            chunk_id += 1

    return {"doc_ids": chunked_doc_ids, "chunk_ids": chunk_ids, text_column_name: chunked_texts, label_column_name: chunked_labels}


if __name__ == "__main__":

        # dataset is you
        max_seq_length = 256
        tokenizer = tokenizer
            
        train_dataset = train_dataset.map(
            lambda x: chunking(x, tokenizer, max_seq_length, "tokens", 'labels'),
            batched=True,
            desc="Splitting documents in chunks for train dataset",
            remove_columns=train_dataset.column_names  
        )

        validation_dataset = validation_dataset.map(
            lambda x: chunking(x, tokenizer, max_seq_length, "tokens", 'labels'),
            batched=True,
            desc="Splitting documents in chunks for validation set",
            remove_columns=validation_dataset.column_names
        )

        test_dataset = test_dataset.map(
            lambda x: chunking(x, tokenizer, max_seq_length, "tokens", 'labels'),
            batched=True,
            desc="Splitting documents in chunks for test set",
            remove_columns=test_dataset.column_names
        )


#Tokenization
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length = 256, padding = 'max_length' )

    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs



#Mapping datasets
tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_validation = validation_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=True)


#Creating Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

#Integrate seqeval
seqeval = evaluate.load('seqeval')

#Compute Metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    

#Early Stopping
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, monitor='eval_loss', patience=3):
        self.monitor = monitor
        self.patience = patience
        self.best_metric = float('inf') if 'loss' in monitor else float('-inf')
        self.counter = 0

    def on_init_end(self, args, state, control, **kwargs):
        pass

    def on_evaluate(self, args, state, control, **kwargs):
        current_metric = state.log_history[-1].get(self.monitor)

        if current_metric is None:
            raise ValueError(f"Metric '{self.monitor}' not found in evaluation results.")

        if self._is_improved(current_metric):
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"No improvement for {self.counter} epochs. Stopping training.")
            control.should_training_stop = True

    def _is_improved(self, metric):
        if 'loss' in self.monitor:
            return metric < self.best_metric
        else:
            return metric > self.best_metric



# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy = "epoch",   # evaluation strategy to adopt during training
    save_strategy = "epoch",
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    num_train_epochs=10,              # total number of training epochs
    weight_decay=0.01,               # strength of weight decay
    push_to_hub=False,               # whether to upload the trained model to the Hub
    logging_dir='./logs',            # directory for storing logs
    logging_steps=500,               # log every n steps
    #load_best_model_at_end=True,     # load the best model when finished training
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_validation,
    compute_metrics = compute_metrics,
    callbacks=[EarlyStoppingCallback(monitor='eval_loss', patience=3)]
)


trainer.train()

# Evaluate the metrics on the test set
test_metrics = trainer.evaluate()

test_metrics
{'eval_loss': 0.0245094932615757,
 'eval_precision': 0.6714709622721492,
 'eval_recall': 0.7674418604651163,
 'eval_f1': 0.716255934885824,
 'eval_accuracy': 0.9926374092391274,
 'eval_runtime': 59.3866,
 'eval_samples_per_second': 62.304,
 'eval_steps_per_second': 0.488,
 'epoch': 6.0}