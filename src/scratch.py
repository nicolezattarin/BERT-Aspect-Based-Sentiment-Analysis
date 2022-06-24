# remove duplicates
def remove_duplicates(data):
    d = data
    rows_drop = []
    for i in range(1,len(d)):
        if d['Tokens'][i] == d['Tokens'][i-1]:
            #sum tags
            tags1 = np.array(d['Tags'][i].strip('][').split(', '), dtype=np.int)
            tags2 = np.array(d['Tags'][i-1].strip('][').split(', '), dtype=np.int)
            pol1 = np.array(d['Polarities'][i].strip('][').split(', '), dtype=np.int)
            pol2 = np.array(d['Polarities'][i-1].strip('][').split(', '),   dtype=np.int)
            tags = tags1+tags2
            tags[tags>1] = 1
            pol = pol1+pol2
            pol[pol>1] = 1
            pol[pol<-1] = -1
            d['Tags'][i] = str([t for t in tags])
            d['Polarities'][i] = str([p for p in pol])
           
            rows_drop.append(i)
            i-=1
    d = d.drop(d.index[rows_drop])
    d.index = range(len(d))
    return d
data_test = remove_duplicates(data_test)


    def test(self, data, load_model=None, device='cpu'):
        
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

        logs_index = int(len(data)/20)
        if logs_index == 0:
            logs_index = 1
        for i in range(len(data)):
            if i % logs_index == 0:
                print('{}/{}'.format(i, len(data)))
                
            sentence = data['Tokens'][i]
            sentence = sentence.replace("'", "").strip("][").split(', ')
            sentence = ' '.join(sentence)
            w, p, _ = self.predict(sentence, load_model=load_model, device=device)
            predictions.append(p)
            tags_real[i] = tags_real[i][:len(p)]
            
        acc = self._accuracy( np.concatenate(tags_real), np.concatenate(predictions))
        return acc, predictions, tags_real