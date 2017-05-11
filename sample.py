#-*- coding:utf-8 -*-
from utils import TextLoader
import tensorflow as tf
import numpy as np 
from model import Model
import os
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
def sample(data,model,args):
    saver=tf.train.Saver()
    with tf.Session() as sess:
        ckpt=tf.train.latest_checkpoint(args.log_dir)
        saver.restore(sess,ckpt)


        prime="今日"
        state=sess.run(model.cell.zero_state(1,tf.float32))

        for word in prime[:-1]:
            x=np.zeros((1,1))
            x[0,0]=data.char2id[word]
            feed={model.input_data:x,model.initial_state:state}
            state=sess.run(model.last_state,feed)

        word=prime[-1]
        lyrics=prime
        for i in range(args.gen_num):
            x=np.zeros([1,1])
            x[0,0]=data.char2id(word)
            feed_dict = {model.input_data: x, model.initial_state: state}
            probs,state=sess.run([model.probs,model.last_state],feed_dict=feed_dict)
            p=probs[0]
            word=data.id2char(np.argmax(p))
            print(word,end='')
            sys.stdout.flush()
            time.sleep(0.05)
            lyrics+=word
        return lyrics




def main():
    tl=TextLoader()
    args=Param()
    lstm_model=Model(args)
    sample(tl,lstm_model,args)


if __name__ == "__main__":
    main()