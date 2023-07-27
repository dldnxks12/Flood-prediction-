import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
from network import *
import os

import re
import random

SOS_token  = 0
EOS_token  = 1
MAX_LENGTH = 20

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        # SOS - 문장의 시작, EOS - 문장의 끝
        self.n_words = 2 # sos, eos 카운트

    def addSentence(self, sentence): # 문장을 단어 단위로 분리, 컨테이너에 추가
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# 데이터 정규화
def normalizeString(df, lang):
    sentence = df[lang].str.lower()  # 소문자 전환
    sentence = sentence.str.replace('[^A-Za-z\s]+', ' ')
    sentence = sentence.str.normalize('NFD')  # 유니코드 정규화
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
    return sentence


def read_sentence(df, lang1, lang2):
    sentence1 = normalizeString(df, lang1)  # 데이터셋 1번째 열
    sentence2 = normalizeString(df, lang2)
    return sentence1, sentence2


def read_file(loc, lang1, lang2):
    df = pd.read_csv(loc, delimiter='\t', header=None, names=[lang1, lang2])
    return df


def process_data(lang1, lang2):
    df = read_file('./%s-%s.txt' % (lang1, lang2), lang1, lang2)  # load data
    sentence1, sentence2 = read_sentence(df, lang1, lang2)

    input_lang  = Lang()
    output_lang = Lang()
    pairs = []
    for i in range(len(df)):
        if len(sentence1[i].split(' ')) < MAX_LENGTH and len(sentence2[i].split(' ')) < MAX_LENGTH:
            full = [sentence1[i], sentence2[i]]    # 1, 2열 합쳐서 저장
            input_lang.addSentence(sentence1[i])   # input으로 영어 사용
            output_lang.addSentence(sentence2[i])  # output으로 프랑스어 사용
            pairs.append(full)  # 입, 출력 합쳐서 사용

    return input_lang, output_lang, pairs

# Tensor로 변환
def indexesFromSentence(lang, sentence): # 문장 분리 및 인덱스 반환
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence): # 문장 끝에 토큰 추가
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair): # 입출력문장 텐서로 변환
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)