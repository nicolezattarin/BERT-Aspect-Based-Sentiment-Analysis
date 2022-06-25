from transformers import BertModel
from transformers import get_scheduler

import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np
import os
from tqdm import tqdm

class ABTEDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokens, tags, pols = self.df.iloc[idx, :3].values

        tokens = tokens.replace("'", "").strip("][").split(', ')
        tags = tags.strip('][').split(', ')
        pols = pols.strip('][').split(', ')

        bert_tokens = []
        bert_tags = []
        bert_pols = []
        for i in range(len(tokens)):
            t = self.tokenizer.tokenize(tokens[i])
            bert_tokens += t
            bert_tags += [int(tags[i])]*len(t)
            bert_pols += [int(pols[i])]*len(t)
        
        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        ids_tensor = torch.tensor(bert_ids)
        tags_tensor = torch.tensor(bert_tags)
        pols_tensor = torch.tensor(bert_pols)
        return bert_tokens, ids_tensor, tags_tensor, pols_tensor

    def __len__(self):
        return len(self.df)

class ABTEBert(torch.nn.Module):
    def __init__(self, pretrain_model, adapter=True):
        super(ABTEBert, self).__init__()
        self.adapter = adapter
        if adapter:
            from transformers.adapters import BertAdapterModel
            self.bert = BertAdapterModel.from_pretrained(pretrain_model)
        else: self.bert = BertModel.from_pretrained(pretrain_model)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, tags_tensors, masks_tensors):
        bert_outputs= self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, return_dict=False)
        bert_outputs = bert_outputs[0]

        linear_outputs = self.linear(bert_outputs)
        if tags_tensors is not None:
            tags_tensors = tags_tensors.view(-1)
            linear_outputs = linear_outputs.view(-1,3)
            loss = self.loss_fn(linear_outputs, tags_tensors)
            return loss
        else:
            return linear_outputs

class ABTEModel ():
    def __init__(self, tokenizer, adapter=True):
        self.model = ABTEBert('bert-base-uncased', adapter=adapter)
        self.tokenizer = tokenizer
        self.trained = False
        self.adapter = adapter

    def padding(self, samples):
        from torch.nn.utils.rnn import pad_sequence
        ids_tensors = [s[1] for s in samples]
        ids_tensors = pad_sequence(ids_tensors, batch_first=True)

        tags_tensors = [s[2] for s in samples]
        tags_tensors = pad_sequence(tags_tensors, batch_first=True)

        pols_tensors = [s[3] for s in samples]
        pols_tensors = pad_sequence(pols_tensors, batch_first=True)
        
        masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)
    
        return ids_tensors, tags_tensors, pols_tensors, masks_tensors

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path), strict=False)
        
    def save_model(self, model, name):
        torch.save(model.state_dict(), name)             

    def train(self, data, epochs, device, batch_size=32, lr=1e-5, load_model=None, lr_schedule=True):

        #load model if lead_model is not None
        if load_model is not None:
            if os.path.exists(load_model):
                self.load_model(self.model, load_model)
                self.trained = True
            else:
                print("lead_model not found")

        # dataset and loader
        ds = ABTEDataset(data, self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=self.padding)
        
        self.model = self.model.to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        num_training_steps = epochs * len(loader)
        if lr_schedule: lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        self.losses = []

        all_data = len(loader)-1
        for epoch in range(epochs):
            finish_data = 0
            current_times = []
            n_batches = int(len(data)/batch_size)

            if self.adapter:
                if lr_schedule: dir_name  = "model_ABTE_adapter_scheduler"
                else: dir_name = "model_ABTE_adapter"
            else:
                if lr_schedule: dir_name  = "model_ABTE_scheduler"
                else: dir_name = "model_ABTE"

            if not os.path.exists(dir_name):
                os.mkdir(dir_name)      

            for nb in range((n_batches)):
                t0 = time.time()

                ids_tensors, tags_tensors, _, masks_tensors = next(iter(loader))
                ids_tensor = ids_tensors.to(device)
                tags_tensor = tags_tensors.to(device)
                masks_tensor = masks_tensors.to(device)
                loss = self.model(ids_tensors=ids_tensor, tags_tensors=tags_tensor, masks_tensors=masks_tensor)
                self.losses.append(loss.item())
                loss.backward()
                optimizer.step()
                if lr_schedule: lr_scheduler.step()
                optimizer.zero_grad()

                finish_data += 1
                current_time = round(time.time() - t0,3)
                current_times.append(current_time)          
                print("epoch: {}\tbatch: {}/{}\tloss: {}\tbatch time: {}\ttotal time: {}"\
                    .format(epoch, finish_data, all_data, loss.item(), current_time, sum(current_times)))
            
                np.savetxt('{}/losses_lr{}_epochs{}_batch{}.txt'.format(dir_name, lr, epochs, batch_size), self.losses)

            self.save_model(self.model, '{}/model_lr{}_epochs{}_batch{}.pkl'.format(dir_name, lr, epoch, batch_size))
            self.trained = True

    def history (self):
        if self.trained:
            return self.losses
        else:
            raise Exception('Model not trained')

    def predict(self, sentence, load_model=None, device='cpu'):
         # load model if exists
        if load_model is not None:
            if os.path.exists(load_model):
                self.load_model(self.model, load_model)
            else:
                raise Exception('Model not found')
        else:
            if not self.trained:
                raise Exception('model not trained')

        word_pieces = list(self.tokenizer.tokenize(sentence))
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        input_tensor = torch.tensor([ids]).to(device)

        #predict
        with torch.no_grad():
            outputs = self.model(input_tensor, None, None)
            _, predictions = torch.max(outputs, dim=2)
            
        predictions = predictions[0].tolist() 
        return word_pieces, predictions, outputs
    
    def predict_batch(self, data, load_model=None, device='cpu'):

        tags_real = [t.strip('][').split(', ') for t in data['Tags']]
        tags_real = [[int(i) for i in t] for t in tags_real]

        # load model if exists
        if load_model is not None:
            if os.path.exists(load_model):
                self.load_model(self.model, load_model)
            else:
                raise Exception('Model not found')
        else:
            if not self.trained:
                raise Exception('model not trained')
        
        predictions = []

        for i in tqdm(range(len(data))):
            sentence = data['Tokens'][i]
            sentence = sentence.replace("'", "").strip("][").split(', ')
            sentence = ' '.join(sentence)
            w, p, _ = self.predict(sentence, load_model=load_model, device=device)
            predictions.append(p)
            tags_real[i] = tags_real[i][:len(p)]
            
        return predictions, tags_real

    def _accuracy (self, x,y):
        return np.mean(np.array(x) == np.array(y))

    def test(self, dataset, load_model=None, device='cpu'):
        from sklearn.metrics import classification_report
        # load model if exists
        if load_model is not None:
            if os.path.exists(load_model):
                self.load_model(self.model, load_model)
            else:
                raise Exception('Model not found')
        else:
            if not self.trained:
                raise Exception('model not trained')

         # dataset and loader
        ds = ABTEDataset(dataset, self.tokenizer)
        loader = DataLoader(ds, batch_size=50, shuffle=True, collate_fn=self.padding)

        pred = []#padded list
        trueth = [] #padded list
        with torch.no_grad():
            for data in tqdm(loader):
                
                ids_tensors, tags_tensors, _, masks_tensors = data
                ids_tensors = ids_tensors.to(device)
                tags_tensors = tags_tensors.to(device)
                masks_tensors = masks_tensors.to(device)

                outputs = self.model(ids_tensors=ids_tensors, tags_tensors=None, masks_tensors=masks_tensors)
                
                _, p = torch.max(outputs, dim=2)

                pred += list([int(j) for i in p for j in i ])
                trueth += list([int(j) for i in tags_tensors for j in i ])
        
        acc = self._accuracy(pred, trueth)
        class_report = classification_report(trueth, pred, target_names=['none', 'start of AT', 'mark of AT'])
        return acc, class_report

    def accuracy(self, data, load_model=None, device='cpu'):
        a, p = self.test(data, load_model=load_model, device=device)
        return a