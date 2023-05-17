import torch
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering, get_linear_schedule_with_warmup)
import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW
import gc

from tqdm.auto import tqdm
from utils import same_seed, read_data, evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# seeding
same_seed(5201314)

# model = AutoModelForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# xlm-roberta-base
# model_name = "bert-base-chinese"
# model_name = "uer/albert-base-chinese-cluecorpussmall"
model_name = "luhua/chinese_pretrain_mrc_macbert_large"
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# preprocessing
# read data
train_questions, train_paragraphs = read_data("data/hw7_train.json")
dev_questions, dev_paragraphs = read_data("data/hw7_dev.json")
test_questions, test_paragraphs = read_data("data/hw7_test.json")

# tokenize
train_questions_tokenized = tokenizer([train_question['question_text'] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question['question_text'] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question['question_text'] for test_question in test_questions], add_special_tokens=False)

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

from datasets import QA_Dataset

train_set = QA_Dataset('train', train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)


from accelerate import Accelerator

# hyperparameters
num_epoch = 20
validation = True
logging_step = 100
learning_rate = 1e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
train_batch_size = 8

# lr_scheduler
warmup_steps = 200
total_steps = len(train_set) * num_epoch

scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

#### TODO: gradient_accumulation (optional)####
# Note: train_batch_size * gradient_accumulation_steps = effective batch size
# If CUDA out of memory, you can make train_batch_size lower and gradient_accumulation_steps upper
# Doc: https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
gradient_accumulation_steps = 16 

# dataloader
# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)


# Change "fp16_training" to True to support automatic mixed 
# precision training (fp16)	
accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=gradient_accumulation_steps)


# Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler) 



model.train()


print("Start Training ...")

for epoch in range(num_epoch):
    step = 1
    train_loss = train_acc = 0
    
    for data in tqdm(train_loader):	
        
        with accelerator.accumulate(model):
            # Load all data into GPU
            # data = [i.to(device) for i in data]
            accelerator.free_memory()
            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
            # Choose the most probable start position / end position
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)
            
            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            
            train_loss += output.loss
            
            accelerator.backward(output.loss)
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1

            # Print training loss and accuracy over past logging step
            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0

    if validation:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
                # prediction is correct only if answer text exactly matches
                dev_acc += evaluate(data, output, tokenizer) == dev_questions[i]["answer_text"]
            print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
        model.train()

# Save a model and its configuration file to the directory 「saved_model」 
# i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
# Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
print("Saving Model ...")
model_save_dir = "saved_model" 
model.save_pretrained(model_save_dir)