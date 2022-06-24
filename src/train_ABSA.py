import os, sys
sys.path.insert(1, '../dataset')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=8, help='batch size')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--lr_schedule', type=bool, default=False, help='learning rate scheduler')
parser.add_argument('--adapter', type=bool, default=True, help='adapter')

def main (batch, epochs, lr, lr_schedule, adapter):

    #load
    data = pd.read_csv('../dataset/normalized/restaurants_train.csv')

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from absa import ABSAModel
    modelABSA = ABSAModel(tokenizer, adapter=True)
    modelABSA.train(data, batch_size=batch, lr=lr, epochs=epochs, device=DEVICE, lr_schedule=True)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.batch, args.epochs, args.lr, args.lr_schedule, args.adapter)
    print('Done')
