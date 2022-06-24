from transformers import BertModel
from transformers import get_scheduler

import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np
import os
from tqdm import tqdm

class ABSADataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokens, tags, pols = self.df.iloc[idx, :3].values

        tokens = tokens.replace("'", "").strip("][").split(', ')
        tags = tags.strip('][').split(', ')
        pols = pols.strip('][').split(', ')

        bert_tokens = []
        bert_att = []
        pols_label = 0
        for i in range(len(tokens)):
            t = self.tokenizer.tokenize(tokens[i])
            bert_tokens += t
            if int(pols[i]) != -1:
                bert_att += t
                pols_label = int(pols[i])

        segment_tensor = [0] + [0]*len(bert_tokens) + [0] + [1]*len(bert_att)
        bert_tokens = ['[cls]'] + bert_tokens + ['[sep]'] + bert_att
        

        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        ids_tensor = torch.tensor(bert_ids)
        pols_tensor = torch.tensor(pols_label)
        segment_tensor = torch.tensor(segment_tensor)

        return bert_tokens, ids_tensor, segment_tensor, pols_tensor

    def __len__(self):
        return len(self.df)

class ABSABert(torch.nn.Module):
    def __init__(self, pretrain_model, adapter=True):
        super(ABSABert, self).__init__()
        self.adapter = adapter
        if adapter:
            from transformers.adapters import BertAdapterModel
            self.bert = BertAdapterModel.from_pretrained(pretrain_model)
        else: self.bert = BertModel.from_pretrained(pretrain_model)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, lable_tensors, masks_tensors, segments_tensors):
        out_dict = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, token_type_ids=segments_tensors)
        linear_outputs = self.linear(out_dict['pooler_output'])

        if lable_tensors is not None:
            loss = self.loss_fn(linear_outputs, lable_tensors)
            return loss
        else:
            return linear_outputs

class ABSAModel ():
    def __init__(self, tokenizer, adapter=True):
        self.model = ABSABert('bert-base-uncased')
        self.tokenizer = tokenizer
        self.trained = False
        self.adapter = adapter

    def padding(self, samples):
        from torch.nn.utils.rnn import pad_sequence
        ids_tensors = [s[1] for s in samples]
        ids_tensors = pad_sequence(ids_tensors, batch_first=True)

        segments_tensors = [s[2] for s in samples]
        segments_tensors = pad_sequence(segments_tensors, batch_first=True)

        label_ids = torch.stack([s[3] for s in samples])
        
        masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)
    
        return ids_tensors, segments_tensors, masks_tensors, label_ids

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

        print ("Training model...")
        print ("Learning rate scheduler: ", lr_schedule)
        print ("Adapter: ", self.adapter)

        ds = ABSADataset(data, self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=self.padding)
        
        self.model = self.model.to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        num_training_steps = epochs * len(loader)

        # possible choices for scheduler are: "constant", "constant_with_warmup", 
        # "polynomial", "cosine_with_restarts", "linear", 'cosine'

        if lr_schedule: lr_scheduler = get_scheduler(name="constant_with_warmup", optimizer=optimizer, 
                                    num_warmup_steps=200, num_training_steps=num_training_steps)

        self.losses = []

        all_data = len(loader)-1

        for epoch in range(epochs):
            finish_data = 0
            current_times = []
            n_batches = int(len(data)/batch_size)

            if self.adapter:
                if lr_schedule: dir_name  = "model_ABSA_adapter_scheduler"
                else: dir_name = "model_ABSA_adapter"
            else:
                if lr_schedule: dir_name  = "model_ABSA_scheduler"
                else: dir_name = "model_ABSA"

            if not os.path.exists(dir_name):
                os.mkdir(dir_name)  

            for nb in range((n_batches)):
                t0 = time.time()

                ids_tensors, segments_tensors, masks_tensors, label_ids = next(iter(loader))
                ids_tensors = ids_tensors.to(device)
                segments_tensors = segments_tensors.to(device)
                label_ids = label_ids.to(device)
                masks_tensors = masks_tensors.to(device)

                loss = self.model(ids_tensors=ids_tensors, lable_tensors=label_ids, 
                                    masks_tensors=masks_tensors, segments_tensors=segments_tensors)
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

    def predict(self, sentence, aspect, load_model=None, device='cpu'):

        #check that the aspect is in the sentence
        if aspect not in sentence:
            raise Exception('Aspect {} not in sentence \n{}'.format(aspect, sentence))

         # load model if exists
        if load_model is not None:
            if os.path.exists(load_model):
                self.load_model(self.model, load_model)
            else:
                raise Exception('Model not found')
        else:
            if not self.trained:
                raise Exception('model not trained')

        t1 = self.tokenizer.tokenize(sentence)
        t2 = self.tokenizer.tokenize(aspect)

        word_pieces = ['[cls]']
        word_pieces += t1
        word_pieces += ['[sep]']
        word_pieces += t2

        segment_tensor = [0] + [0]*len(t1) + [0] + [1]*len(t2)

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        input_tensor = torch.tensor([ids]).to(device)
        segment_tensor = torch.tensor(segment_tensor).to(device)

        with torch.no_grad():
            outputs = self.model(input_tensor, None, None, segments_tensors=segment_tensor)
            _, predictions = torch.max(outputs, dim=1)
        
        return word_pieces, int(predictions)-1, outputs

    def _accuracy (self, x,y):
        acc = 0
        for i in range(len(x)):
            if x[i] == y[i]:
                acc += 1
        return acc/len(x)

    def predict_batch(self, data, load_model=None, device='cpu'):
        
        tags_real = [t.strip('][').split(', ') for t in data['Tags']]
        tags_real = [[int(i) for i in t ] for t in tags_real]
        
        polarity_real = [t.strip('][').split(', ') for t in data['Polarities']]
        # if -1 is not an aspect term, if 0 negative, if 2 positive, if 1 neutral, shift of 1
        polarity_real = [[int(i)-1 if int(i)>-1 else None for i in t ] for t in polarity_real]

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
            sentenceList = sentence.replace("'", "").strip("][").split(', ')
            sentence = ' '.join(sentenceList)
            prediction = []
            for j in range(len(sentenceList)):
                if tags_real[i][j] != 0:
                    aspect = sentenceList[j]

                    w, p, _ = self.predict(sentence, aspect, load_model=load_model, device=device)
                    prediction.append(p)
                else:
                    prediction.append(None)
            predictions.append(prediction)
            polarity_real[i] = polarity_real[i][:len(prediction)]
        return predictions, polarity_real

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
        ds = ABSADataset(dataset, self.tokenizer)
        loader = DataLoader(ds, batch_size=50, shuffle=True, collate_fn=self.padding)

        pred = []#padded list
        trueth = [] #padded list
     
        with torch.no_grad():
            for data in tqdm(loader):
            
                ids_tensors, segments_tensors, masks_tensors, label_ids = data
                ids_tensors = ids_tensors.to(device)
                segments_tensors = segments_tensors.to(device)
                masks_tensors = masks_tensors.to(device)

                outputs = self.model(ids_tensors, None, masks_tensors=masks_tensors, 
                                    segments_tensors=segments_tensors)
                
                _, p = torch.max(outputs, dim=1)   
                pred += list([int(i) for i in p])
                trueth += list([int(i) for i in label_ids])    
        acc = self._accuracy(pred, trueth)
        class_report = classification_report(trueth, pred, target_names=['negative', 'neutral', 'positive'])
        return acc, class_report
        
    def accuracy(self, data, load_model=None, device='cpu'):
        a, p = self.test(data, load_model=load_model, device=device)
        return a