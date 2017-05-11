import os
import sys
import tensorflow as tf
import numpy as np 
from utils import TextLoader
from model import Model
# args.batch_size 
#         args.state_size  
#         args.num_layers
#         args.seq_length
#         args.rnn_size
#         args.vocab_size

class Param():
    batch_size = 128
    state_size = 100
    num_layers = 3
    seq_length = 50
    rnn_size = state_size
    vocab_size = 3690
    grad_clip = 5
    decay_rate = 0.9
    learning_rate = 0.002
    num_epoches = 10
    log_dir = './logs'







def train(data,model,args):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        for i in range(args.num_epoches):
            sess.run(tf.assign(model.lr,args.learning_rate*(args.decay_rate**i)))
            data.pointer = 0
            for b in range(data.get_numbatches()):
                x,y=data.next_batch()
                feed = {model.input_data:x,model.targets:y}
                train_loss, _ , _ = sess.run([model.cost, model.last_state, model.train_op], feed)
                print("epoch:{}   train_loss:{}".format(i,train_loss))
            if b % 50 == 0 or (b+1) == data.get_numbatches():
                saver.save(sess,os.path.join(args.log_dir,"poem_model.ckpt"),global_step=i*data.get_numbatches()+b)


        




def main():
    tl=TextLoader()
    args=Param()
    lstm_model=Model(args)
    train(tl,lstm_model,args)


if __name__ == "__main__":
    main()