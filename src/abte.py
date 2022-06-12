from transformers import BertModel
import torch
from torch.utils.data import Dataset
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os,sys

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
    def __init__(self, pretrain_model):
        super(ABTEBert, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
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
    def __init__(self, tokenizer):
        self.model = ABTEBert('bert-base-uncased')
        self.tokenizer = tokenizer
        self.trained = False

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
                

    def train(self, data, epochs, device, batch_size=32, lr=1e-5):

        # dataset and loader
        ds = ABTEDataset(data, self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=self.padding)
        
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.losses = []

        all_data = len(loader)-1
        for epoch in range(epochs):
            finish_data = 0
            current_times = []

            n_batches = int(len(data)/batch_size)
            # batch = next(iter(loader))
            # print (batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape)
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
                optimizer.zero_grad()

                finish_data += 1
                current_time = round(time.time() - t0,3)
                current_times.append(current_time)
                print("epoch: {}\tbatch: {}/{}\tloss: {}\tbatch time: {}\ttotal time: {}"\
                    .format(epoch, finish_data, all_data, loss.item(), current_time, sum(current_times)))
                np.savetxt('losses_lr{}_epochs{}_batch{}.txt'.format(lr, epochs, batch_size), self.losses)

            self.save_model(self.model, 'model_lr{}_epochs{}_batch{}.pkl'.format(lr, epoch, batch_size))
            self.trained = True

    def history (self):
        if self.trained:
            return self.losses
        else:
            raise Exception('Model not trained')

    def unpack_sequence(self, packed_sequence, mask):
        unpacked_sequence = []
        for i in range(len(packed_sequence)):
            if mask[i] == 1:
                unpacked_sequence.append(packed_sequence[i])
    
        return unpacked_sequence

    def predict(self, sentence, device='cpu', batch_size=256, lr=1e-4, epochs=2):
         # load model if exists
        if os.path.exists('model_lr{}_epochs{}_batch{}.pkl'.format(lr, epochs, batch_size)):
            self.load_model(self.model, 'model_lr{}_epochs{}_batch{}.pkl'.format(lr, epochs, batch_size))
        if not self.trained and not os.path.exists('model_lr{}_epochs{}_batch{}.pkl'.format(lr, epochs, batch_size)):
            raise Exception('model not trained and does not exist')

        word_pieces = []
        tokens = self.tokenizer.tokenize(sentence)
        word_pieces += tokens

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        input_tensor = torch.tensor([ids]).to(device)

        with torch.no_grad():
            outputs = self.model(input_tensor, None, None)
            _, predictions = torch.max(outputs, dim=2)
        predictions = predictions[0].tolist()

        return word_pieces, predictions, outputs

    def _accuracy (self, x,y):
        acc = 0
        for i in range(len(x)):
            if x[i] == y[i]:
                acc += 1
        return acc/len(x)

    def test(self, data, device='cpu', batch_size=256, lr=1e-4, epochs=2):
        
        tags_real = [t.strip('][').split(', ') for t in data['Tags']]
        tags_real = [[int(i) for i in t] for t in tags_real]

        # load model if exists
        if os.path.exists('model_lr{}_epochs{}_batch{}.pkl'.format(lr, epochs, batch_size)):
            self.load_model(self.model, 'model_lr{}_epochs{}_batch{}.pkl'.format(lr, epochs, batch_size))
        if not self.trained and not os.path.exists('model_lr{}_epochs{}_batch{}.pkl'.format(lr, epochs, batch_size)):
            raise Exception('model not trained and does not exist')
        
        predictions = []
        for i in range(len(data)):
            sentence = data['Tokens'][i]
            sentence = sentence.replace("'", "").strip("][").split(', ')
            sentence = ' '.join(sentence)
            w, p, _ = self.predict(sentence, device, batch_size, lr, epochs)
            predictions.append(p)
        acc = self._accuracy( np.concatenate(tags_real), np.concatenate(predictions))
        return acc, predictions, tags_real

    def accuracy(self, data, device='cpu', batch_size=256, lr=1e-4, epochs=2):
        a, p = self.test(data, device, batch_size, lr, epochs)
        return a