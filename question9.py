import os
import time

import torch
import random
import numpy as np
from torch import optim
from tqdm import tqdm

import torch
import torch.nn as nn

from torchtext.legacy import data
from IPython.display import HTML

SEED = 2022
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)  # 为CPU设置随机种子
torch.cuda.manual_seed(SEED)  # 为GPU设置随机种子
torch.backends.cudnn.deterministic = True  # 可以提升一点训练速度，没有额外开销。


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):  # text=[seq_len, batch_size]
        embedded = self.dropout(self.embedding(text))  # [seq_len, batch_size, embedding_dim]
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden.squeeze(0))

def binary_accuracy(preds, y):
    '''计算准确度，即预测和实际标签的相匹配的个数'''
    rounded_preds = torch.round(torch.sigmoid(preds))  # .round函数：四舍五入[neg: 0, pos: 1]
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, loss_fn):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0
    model.train()  # 训练时会用到dropout、归一化等方法，但测试的时候不能用dropout等方法

    for batch in iterator:
        print(batch.text[5])
        optimizer.zero_grad()
        preds = model(batch.text).squeeze(1)  # squeeze(1)压缩维度，和batch.label维度对上
        loss = loss_fn(preds, batch.label)
        acc = binary_accuracy(preds, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(batch.label)

        epoch_acc += acc.item() * len(batch.label)

        total_len += len(batch.label)
        break

    return epoch_loss / total_len, epoch_acc / total_len


def evaluate(model, iterator, loss_fn):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0
    model.eval()
    # 转换成测试模式，冻结dropout层或其他层。

    with torch.no_grad():
        print(type(iterator))
        for batch in iterator:
            preds = model(batch.text).squeeze(1)
            loss = loss_fn(preds, batch.label)
            acc = binary_accuracy(preds, batch.label)

            epoch_loss += loss.item() * len(batch.label)
            epoch_acc += acc.item() * len(batch.label)
            total_len += len(batch.label)
    model.train()
    return epoch_loss / total_len, epoch_acc / total_len

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def create_download_link(title = "Download model file", filename = "wordavg-model.pt"):
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title, filename=filename)
    return HTML(html)

if __name__ == '__main__':
    TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
    LABEL = data.LabelField(dtype=torch.float)

    train_data, val_data = data.TabularDataset.splits(
        path='./data', train='train.csv', validation='valid.csv', format='csv', skip_header=True,
        fields=[('text', TEXT), ('label', LABEL)])

    test_data = data.TabularDataset('./data/test.csv', format='csv', skip_header=True,fields=[('text', TEXT), ('label', LABEL)])

    TEXT.build_vocab(train_data, max_size=5000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data)

    BATCH_SIZE = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 相当于把样本划分batch，把相等长度的单词尽可能的划分到一个batch，不够长的就用padding。
    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        device=device)

    VOCAB_SIZE = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

    pretrained_embeddings = TEXT.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)

    N_EPOCHS = 20
    best_valid_loss = float('inf')

    for epoch in tqdm(range(N_EPOCHS)):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        print('hhh')
        valid_loss, valid_acc = evaluate(model, val_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'lstm-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    create_download_link(filename='lstm-model.pt')
    model.load_state_dict(torch.load('lstm-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')