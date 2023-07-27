import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os

import re
import random

SOS_token  = 0
EOS_token  = 1
MAX_LENGTH = 20

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Seq2seq 네트워크
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()
        # 인코더와 디코더 초기화
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_lang, output_lang, teacher_forcing_ratio=0.5):

        input_length  = input_lang.size(0)  # 입력 문장 길이(문장 단어수)
        batch_size    = output_lang.shape[1]
        target_length = output_lang.shape[0]
        vocab_size    = self.decoder.output_dim
        outputs       = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        for i in range(input_length):
            # 문장의 모든 단어 인코딩
            encoder_output, encoder_hidden = self.encoder(input_lang[i])

        # 인코더 은닉층 -> 디코더 은닉층
        decoder_hidden = encoder_hidden.to(device)
        # 예측 단어 앞에 SOS token 추가
        decoder_input = torch.tensor([SOS_token], device=device)

        for t in range(target_length):
            # 현재 단어에서 출력단어 예측
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            # teacher force 활성화하면 모표를 다음 입력으로 사용
            input = (output_lang[t] if teacher_force else topi)
            # teacher force 활성화하지 않으면 자체 예측 값을 다음 입력으로 사용
            if (teacher_force == False and input.item() == EOS_token):
                break
        return outputs


# 인코더 네트워크
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim    # 인코더 입력층
        self.embbed_dim = embbed_dim  # 인코더 임베딩 계층
        self.hidden_dim = hidden_dim  # 인코더 은닉층(이전 은닉층)
        self.num_layers = num_layers  # GRU 계층 개수
        self.embedding = nn.Embedding(input_dim, self.embbed_dim)  # 임베딩 계층 초기화
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        # 임베딩 차원, 은닉층 차원, gru 계층 개수를 이용하여 gru 계층 초기화

    def forward(self, src):
        embedded = self.embedding(src).view(1, 1, -1)  # 임베딩
        outputs, hidden = self.gru(embedded)  # 임베딩 결과를 GRU 모델에 적용
        return outputs, hidden


# 디코더 네트워크
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()

        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, self.embbed_dim)  # 임베딩 초기화
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)  # gru 초기화
        self.out = nn.Linear(self.hidden_dim, output_dim)  # 선형 계층 초기화
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden