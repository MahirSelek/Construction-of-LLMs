#Import Libraries
import os
import wandb
wandb.login(key="login_key")
wandb.init(project='project_name', name='specific_name')


import torch
from transformers import AutoModelForMaskedLM, BertForMaskedLM
from transformers import AutoTokenizer, BertTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertForTokenClassification

from datasets import concatenate_datasets
from datasets import load_dataset



train1 = load_dataset('json', data_files = 'FOLD_1.json' )
train2 = load_dataset('json', data_files = 'FOLD_2.json' )
train3 = load_dataset('json', data_files = 'FOLD_3.json' )
#silver_dataset = load_dataset('json', data_files = '/home/mselek/Ner/silver.json' )

validation_dataset = load_dataset('json', data_files = 'FOLD_4.json' )
test_dataset = load_dataset('json', data_files = 'FOLD_5.json' )

# %%

validation_dataset = validation_dataset['train']
test_dataset = test_dataset['train']


#Remove Labels
train_dataset1 = train1.remove_columns("labels")
train_dataset2 = train2.remove_columns("labels")
train_dataset3 = train3.remove_columns("labels")
#silver_dataset = silver_dataset.remove_columns('labels')

validation_dataset = validation_dataset.remove_columns("labels")
test_dataset = test_dataset.remove_columns("labels")

#Merge Train and Silver dataset
merged_dataset = concatenate_datasets([train_dataset1['train'], train_dataset2['train'], train_dataset3['train']])


#Create Model and Tokenizer
#MODEL_NAME = "bert-base-uncased"
#model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space = True, model_max_length=256)


model_directory ='bert-base-uncased'
model = BertForMaskedLM.from_pretrained(model_directory)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', add_prefix_space = True, model_max_length=512)



#Create Chunks
def get_chunk_truncation_size(doc_txt, tokenizer, max_seq_length=512):
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


def chunking(examples, tokenizer, seq_max_length, text_column_name=None):
    # split of wide pages in 256 word level chunks
    text_column_name = text_column_name if text_column_name else "tokens"
    

    texts = examples[text_column_name]
    doc_ids = examples["document_id"]
    chunked_texts, chunked_doc_ids, chunk_ids = [], [], []
    for i in range(len(texts)):
        doc_text = texts[i]

        residual_text = doc_text

        doc_id = doc_ids[i]
        chunk_id = 0
        while len(residual_text) > 0:

            chunk_ids.append(chunk_id)
            chunk_len = get_chunk_truncation_size(
                residual_text,
                tokenizer,
                seq_max_length
            )

            chunked_texts.append(residual_text[:chunk_len])
            chunked_doc_ids.append(doc_id)

            residual_text = residual_text[chunk_len:]  
            chunk_id += 1        

    return {"doc_ids": chunked_doc_ids, "chunk_ids": chunk_ids, text_column_name: chunked_texts}

if __name__ == "__main__":

    max_seq_length = 512
    tokenizer = tokenizer
    dataset = merged_dataset
    
    dataset = dataset.map(
        lambda x: chunking(x, tokenizer, max_seq_length, "tokens"),
        batched=True,
        desc="Splitting documents in chunks for merged dataset",
        remove_columns=dataset.column_names  
    )

    validation_dataset = validation_dataset.map(
        lambda x: chunking(x, tokenizer, max_seq_length, "tokens"),
        batched=True,
        desc="Splitting documents in chunks for validation set",
        remove_columns=validation_dataset.column_names
    )

    test_dataset = test_dataset.map(
        lambda x: chunking(x, tokenizer, max_seq_length, "tokens"),
        batched=True,
        desc="Splitting documents in chunks for test set",
        remove_columns=test_dataset.column_names
    )


    #Tokenization after chunking
    def tokenization(example):
        return tokenizer(example['tokens'], is_split_into_words=True, truncation=True, padding="max_length")


    #Mapping on dataset
    dataset = dataset.map(tokenization, batched=True)
    validation_dataset = validation_dataset.map(tokenization, batched=True)
    test_dataset = test_dataset.map(tokenization, batched=True)


    #Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True , mlm_probability=0.15
    )


    #Training Arguments
    training_args = TrainingArguments(
        output_dir='./mlm-vt-roberta',
        evaluation_strategy='epoch',
        #eval_steps=1000,
        #save_steps=1000,
        #set_save="epoch",
        save_strategy = 'epoch',
        num_train_epochs=20, 
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8, 
        learning_rate=1e-4, 
        warmup_steps=1000,
        weight_decay=0.01,
	report_to = 'wandb'
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=validation_dataset
    )


    #Train
    training_output = trainer.train()

    print(training_output)

    #Save Pretrained Model
    trainer.save_model("./mlm_vt_bert")
