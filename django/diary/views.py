# -*- coding: utf-8 -*-
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from .models import User, Result, Content
from .forms import ContentForm, UserForm, ResultForm
import os
import re

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from tqdm.notebook import tqdm

from kobert.utils import get_tokenizer 
from kobert.pytorch_kobert import get_pytorch_kobert_model 

from transformers import AdamW 
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

module_dir = os.path.dirname(__file__)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=4,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

def index(request):
    return render(request, 'diary/diary.html')


def analysis(request):
    if request.method == 'POST':
        data = request.read().decode('utf-8')
        #text = erase_hex(data)
        text = data_preprocess(data)
        print(data)
        file = os.path.join(module_dir, 'kobert_ending2.pt')
        try:
            model = load_model(file)
        except:
            model = run_model()
        
        result = predict(model, text)
        results = calc_result(result)
        
        #if form.is_valid():
            #form.save()
    
        return JsonResponse({"results":results})



def erase_hex(data):
    texts = []
    for text in data:
        tmp = text[0].replace('&#x27;','')
        texts.append(tmp, text[1])
    
    return texts

def result(request):
    if request.method == 'POST':
        form = ContentForm(request.POST)
        data = form.data['text']
        
        return render(request, 'diary/result.html', {'text':data})

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def run_model():
    device = torch.device("cuda:0")
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    
    max_len = 64
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 2
    max_grad_norm = 1
    log_interval = 200
    learning_rate =  5e-5
    
    dataset_train, dataset_test = train_test_split(dtls, test_size=0.2, random_state=123)
    
    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)
    

    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
    
    #��Ƽ�������� �ս��Լ� ����
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
    
    
    #�н� ����
    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device) 
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1))) 
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1))) 
        model.eval() #�� �� �κ� 
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
        
        
    return model
    
def load_model(file):
    device = torch.device("cuda:0")
    bertmodel, vocab = get_pytorch_kobert_model()
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
    model.load_state_dict(torch.load(file))
    model.eval()
    
    return model
    
def data_preprocess(data):
    raw = re.split('[\r\n\.\?\!]', data)
    #raw = data.replace('\r\n', ' ').replace('.', ' ').replace('\?', ' ')
    text = []
    
    for val in raw:
        if val == '':
            continue
        text.append([val, 0.0])
    
    
    print(text)
    return text

def predict(model, text):
    device = torch.device("cuda:0")
    max_len = 64
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 2
    max_grad_norm = 1
    log_interval = 200 
    learning_rate =  5e-5
    
    tokenizer = get_tokenizer()
    bertmodel, vocab = get_pytorch_kobert_model()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    data_test = BERTDataset(text, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)
    model.eval()
    
    answer=[]
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        max_vals, max_indices = torch.max(out, 1)
        answer.append(max_indices.cpu().clone().numpy())
    
    result = F.softmax(out)
    
    print(result)
    return result
    
def calc_result(result):
    sadness = 0.0
    joy = 0.0
    anger = 0.0
    fear = 0.0
    result = result.detach().cpu().clone().numpy()
    
    for data in result:
        sadness += data[0]
        joy += data[1]
        anger += data[2]
        fear += data[3]
    
    result = [sadness, joy, anger, fear]
    
    
    
    results = normalize(result)
    results = [0.1 if x == 0.0 else x for x in results]
    results = {'sadness': results[0], 'joy': results[1], 'anger': results[2], 'fear': results[3]}
    
    return results

def normalize(result):
    max_ = max(result)
    min_ = min(result)
    list = []
    for val in result:
        val = (val - min_)/(max_ - min_)
        list.append(round(val,2))

    print(list)
    return list
    
    
    
    
    
    
    
    