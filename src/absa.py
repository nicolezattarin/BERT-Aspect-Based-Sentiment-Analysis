from transformers import BertModel
import torch
from torch.utils.data import Dataset
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os,sys
    
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
    def __init__(self, pretrain_model):
        super(ABSABert, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, lable_tensors, masks_tensors, segments_tensors):
        _, pooled_outputs = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, token_type_ids=segments_tensors)
        linear_outputs = self.linear(pooled_outputs)

        if lable_tensors is not None:
            loss = self.loss_fn(linear_outputs, lable_tensors)
            return loss
        else:
            return linear_outputs


class ABTEModel ():
    def __init__(self, tokenizer):
        self.model = ABSABert('bert-base-uncased')
        self.tokenizer = tokenizer
        self.trained = False

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
                

    def train(self, data, epochs, device, batch_size=32, lr=1e-5):

        # dataset and loader
        ds = ABSADataset(data, self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=self.padding)
        
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        all_data = len(loader)-1
        self.losses = []

        for epoch in range(epochs):
            finish_data = 0
            current_times = []

            n_batches = int(len(data)/batch_size)
            # batch = next(iter(loader))
            # print (batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape)
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

    def predict(self, sentence, aspect, device='cpu', batch_size=256, lr=1e-4, epochs=2):
         # load model if exists
        if os.path.exists('model_lr{}_epochs{}_batch{}.pkl'.format(lr, epochs, batch_size)):
            self.load_model(self.model, 'model_lr{}_epochs{}_batch{}.pkl'.format(lr, epochs, batch_size))
        if not self.trained and not os.path.exists('model_lr{}_epochs{}_batch{}.pkl'.format(lr, epochs, batch_size)):
            raise Exception('model not trained and does not exist')

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