import os
import sys
import time


import numpy as np 
import tensorflow as tf 
from tensorflow.python.ops import seq2seq,rnn_cell 

class Model():
    def __init__(self,args,infer=False):
        self.args = args
        if infer == True:
            args.batch_size = 1
            args.seq_length = 1
        #

        cell = rnn_cell.BasicLSTMCell(args.state_size)  #
        self.cell = cell = rnn_cell.MultiRNNCell([cell]*args.num_layers)

        self.input_data = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
        self.targets = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size,tf.float32)

        with tf.variable_scope('rnnlm'):
            w = tf.get_variable('softmax_w',[args.rnn_size,args.vocab_size])
            b = tf.get_variable('softmax_b',[args.vocab_size])

            with tf.device('/cpu:0'):
                embedding = tf.get_variable('embedding',[args.vocab_size,args.rnn_size])
                inputs= tf.nn.embedding_lookup(embedding,self.input_data)

        outputs,last_state = tf.nn.dynamic_rnn(self.cell,inputs,initial_state=self.initial_state,scope='rnnlm')
        output = tf.reshape(outputs,[-1,args.rnn_size])

        self.logits = tf.matmul(output,w)+b
        self.probs = tf.nn.softmax(self.logits)
        targets=tf.reshape(self.targets,[-1])
        loss = seq2seq.sequence_loss_by_example([self.logits],[targets],[tf.ones_like(targets,dtype=tf.float32)])
        self.cost = tf.reduce_mean(loss)

        self.last_state=last_state


        self.lr = tf.Variable(0.0,trainable=False)#
        optimizer=tf.train.AdamOptimizer(self.lr)
        tvars=tf.trainable_variables()
        grads=tf.gradients(self.cost,tvars)
        grads,_=tf.clip_by_global_norm(grads,args.grad_clip)
        self.train_op=optimizer.apply_gradients(zip(grads,tvars))



