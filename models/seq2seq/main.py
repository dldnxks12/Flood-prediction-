import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os

import re
import random
from network import *
from utils import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

SOS_token  = 0
EOS_token  = 1
MAX_LENGTH = 20

teacher_forcing_ratio = 0.5

def Model(model, input_tensor, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    output = model(input_tensor, target_tensor)
    num_iter = output.size(0)

    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])

    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter
    return epoch_loss


# 모델 훈련함수 정의
def trainModel(model, input_lang, output_lang, pairs, num_iteration=20000):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0

    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
                      for i in range(num_iteration)]

    for iter in range(1, num_iteration + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = Model(model, input_tensor, target_tensor, optimizer, criterion)
        total_loss_iterations += loss

        if iter % 5000 == 0:
            average_loss = total_loss_iterations / 5000
            total_loss_iterations = 0
            print('%d %.4f' % (iter, average_loss))

    torch.save(model.state_dict(), './mytraining.pt')
    return model


# 모델 평가
def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentences[0])
        output_tensor = tensorFromSentence(output_lang, sentences[1])
        decoded_words = []
        output = model(input_tensor, output_tensor)

        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)

            if topi[0].item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi[0].item()])
    return decoded_words


def evaluateRandomly(model, input_lang, output_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('input {}'.format(pair[0]))
        print('output {}'.format(pair[1]))
        output_words = evaluate(model, input_lang, output_lang, pair)
        output_sentence = ' '.join(output_words)
        print('predicted {}'.format(output_sentence))


if __name__ == '__main__':
    lang1 = 'eng'
    lang2 = 'fra'
    input_lang, output_lang, pairs = process_data(lang1, lang2)

    randomize = random.choice(pairs)
    print('random sentence {}'.format(randomize))

    input_size = input_lang.n_words
    output_size = output_lang.n_words
    print('Input : {} Output : {}'.format(input_size, output_size))

    embed_size = 256
    hidden_size = 512
    num_layers = 1
    num_iteration = 75000

    encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
    decoder = Decoder(output_size, hidden_size, embed_size, num_layers)

    model = Seq2Seq(encoder, decoder, device).to(device)

    print(encoder)
    print(decoder)

    model = trainModel(model, input_lang, output_lang, pairs, num_iteration)