import os
import sys
import time
from six.moves import cPickle

import numpy as np
import tensorflow as tf 

class TextLoader():
    def __init__(self,batch_size = 128,seq_length = 50,num_epoches = 10):
        self.batch_size=batch_size
        self.seq_length = seq_length
        self.num_epoches = num_epoches
        self.vocab_file="vocab.pkl"
        with open("1.txt","r",encoding="utf-8",errors='ignore') as f:
            self.data=f.read()
        self.total_len =len(self.data)
        self.words = list(set(self.data))
        self.words.sort()
        self.pointer = 0
        self.vocab_size=len(self.words)
        print("Vocab_size: ",self.vocab_size)


        self.char2id = { w : i for i,w in enumerate(self.words)}
        self.id2char = { i : w for i,w in enumerate(self.words)}

        with open(self.vocab_file,"wb" ) as f:
            cPickle.dump(self.words,f)
        self.num_batches = self.total_len//(self.batch_size*self.seq_length)
        

    def next_batch(self):
        x_batches = []
        y_batches = []
        for i in range(self.batch_size):
            if self.pointer + self.seq_length +1 >=self.total_len:
                self.pointer = 0
            bx = self.data[self.pointer : self.pointer+self.seq_length]
            by = self.data[self.pointer + 1 : self.pointer + self.seq_length + 1]
            self.pointer += self.seq_length

            bx = [self.char2id[c] for c in bx]
            by = [self.char2id[c] for c in by]

            x_batches.append(bx)
            y_batches.append(by)


        return x_batches,y_batches 


    def get_dict(self):
        return self.char2id,self.id2char

    def get_vocabsize(self):
        return self.vocab_size

    def get_numbatches(self):
        return self.num_batches

