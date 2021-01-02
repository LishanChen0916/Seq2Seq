from __future__ import unicode_literals, print_function, division
import math
import random
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from dataloader import *

'''Hyper parameters'''
# 26 alphabet plus "SOS" and "EOS"
letter_size = 28
hidden_size = 256
learning_rate = 0.05
'''Hyper parameters'''


# Compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs, hidden):
        embedded = self.dropout(self.embedding(inputs)).view(-1, 1, self.hidden_size)

        output, (hidden_, cell_) = self.lstm(embedded, hidden)
        # output shape = [seq_len, batch_size, hidden_size]
        # hidden_ shape = [1, 1, hidden_size]
        # c_ shape = [1, 1, hidden_size]
        
        # We just need to return hidden(final hidden state) and cell(final cell state)
        return (hidden_, cell_)

    def initHidden(self):
        h0 = torch.zeros(1, 1, self.hidden_size, device=device)
        c0 = torch.zeros(1, 1, self.hidden_size, device=device)
        return (h0, c0)

# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(hidden_size, letter_size)

    def forward(self, inputs, hidden):
        # As we are only decoding one token at a time, the seq_len will always equal to 1
        embedded = self.dropout(self.embedding(inputs)).view(1, 1, self.hidden_size)
        output = F.relu(embedded)
        
        output, (hidden_, cell_) = self.lstm(output, hidden)
        pred = self.out(output).view(-1, letter_size)

        return pred, (hidden_, cell_)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden_size of encoder and decoder must be equal!"
        
    def forward(self, inputs, target, teacher_forcing_ratio=0.5):
        target_len = target.shape[0]

        # Store decoder's outputs
        outputs = torch.zeros(target_len, 1, letter_size).to(device)
        hidden_, cell_ = self.encoder(inputs, self.encoder.initHidden())

        # Decoder's first input: 'SOS' token
        input_ = torch.unsqueeze(target[0], 0)
        prediction = []

        for t in range(1, target_len):
            output, (hidden_, cell_) = self.decoder(input_, (hidden_, cell_))
            
            # Place predictions in a tensor for each token
            outputs[t] = output

            # Decide whether we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            pred = output.argmax(1)
            input_ = torch.unsqueeze(target[t], 0) if teacher_force else pred

            if pred != 27:
                prediction.append(pred)

        return outputs, prediction


def train(train_loader, test_loader, model, epochs=300, optimizer=optim.SGD, criterion=nn.CrossEntropyLoss()):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_ = np.zeros(epochs)
    bleu_score = []
    bleu_score_ = np.zeros(epochs)

    for epoch in range(epochs):
        loss_sum = 0
        model.train()
        for train_data, train_label in train_loader:
            # Remove all the dimensions of input of size 1
            train_data = torch.squeeze(train_data).to(device)
            train_label = torch.squeeze(train_label).to(device)

            optimizer.zero_grad()
            
            output, prediction = model(train_data, train_label)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            target = train_label[1:].view(-1)

            loss = criterion(output, target)
            loss_sum += loss
            loss_[epoch] = loss_sum
            loss.backward()
            optimizer.step()

        print("Epochs[%3d/%3d] Loss : %f" % (epoch, epochs, loss_sum))

        model.eval()
        with torch.no_grad():
            for test_data, test_label in test_loader:
                test_data = torch.squeeze(test_data).to(device)
                test_label = torch.squeeze(test_label).to(device)

                output, prediction = model(test_data, test_label, 0)
                
                targetString = c.LongtensorToString(test_label, show_token=False, check_end=True)
                outputString = c.LongtensorToString(prediction, show_token=False, check_end=False)
                bleu_score.append(compute_bleu(outputString, targetString))
                bleu_score_[epoch] = sum(bleu_score) / len(bleu_score)

            print('BlEU-4 Score: {score}'.format(score=sum(bleu_score) / len(bleu_score)))

    torch.save(model.state_dict(), 'Seq2Seq_Dropout.pkl')
    plotCurve('Training: Loss&Score', loss_, bleu_score_)
    

def evaluate(test_loader, model, path):
    bleu_score = []
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        for test_data, test_label in test_loader:
            test_data = torch.squeeze(test_data).to(device)
            test_label = torch.squeeze(test_label).to(device)

            output, prediction = model(test_data, test_label, 0)
            print('=========================')
            print('Input:  ', c.LongtensorToString(test_data, show_token=False, check_end=True))
            print('Target: ', c.LongtensorToString(test_label, show_token=False, check_end=True))
            print('Pred:   ', c.LongtensorToString(prediction, show_token=False, check_end=False))
            print('=========================')

            targetString = c.LongtensorToString(test_label, show_token=False, check_end=True)
            outputString = c.LongtensorToString(prediction, show_token=False, check_end=False)
            bleu_score.append(compute_bleu(outputString, targetString))

        print('BlEU-4 Score: {score}'.format(score=sum(bleu_score) / len(bleu_score)))


def plotCurve(title, loss, score):
    plt.figure(figsize=(8,4.5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss, label='Loss')
    h1, l1 = plt.gca().get_legend_handles_labels()

    ax = plt.gca().twinx()
    plt.ylabel('Score')
    ax.plot(score, label='Score', c='C2')
    h2, l2 = ax.get_legend_handles_labels()

    plt.legend(loc='best')
    plt.title(title)

    ax.legend(h1+h2, l1+l2, loc='best')

    #plt.show()
    filename = title
    plt.savefig(filename + ".png")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = JsonDataloader('Data', 'train')
    test_dataset = JsonDataloader('Data', 'test')

    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)

    c = CharDict()
    encoder = EncoderRNN(input_size=letter_size, hidden_size=hidden_size).to(device)
    decoder = DecoderRNN(hidden_size=hidden_size, output_size=letter_size).to(device)

    model = Seq2Seq(encoder, decoder).to(device)

    # Train
    train(train_loader, test_loader, model)

    # Evaluate
    path = 'Result/Seq2Seq_Dropout.pkl'
    evaluate(test_loader, model, path)
