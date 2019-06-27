import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


_N_W = 2
_N_B = 8
_N_A = 8

#from model import simple_mlp
from model_fixed import simple_mlp


def main():
    #import data
    dataset = input_data.read_data_sets('./data/', one_hot = False)


    #create parameters
    model = simple_mlp(
            n_layer = 3,
            hidden_dim = 512
            )


    #placeholder for batch data
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.int32, [None,])

    #build network graph
    logit = model(x, n_w = _N_W, n_b = _N_B, n_a = _N_A)
    
    #define loss and optimizers
    loss = tf.losses.sparse_softmax_cross_entropy(labels = y, logits = logit) 
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(tf.argmax(logit, axis=-1)), y)))

    saver = tf.train.Saver()
    ###### end graph build ######


    ### call graph through tf.Session() ###
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, './results/model')
    

    ### test trained model ###
    test_loss, test_acc = sess.run(
            [loss, acc],
            feed_dict = {x: dataset.test.images, y: dataset.test.labels}
            )

    print('test loss : %.4f, acc : %.4f'%(test_loss, test_acc))
    
    all_p = sess.run(tf.global_variables())
    
    
    #check parameters
    print('-'*50)
    print('float parmas')
    for p, pp in zip(all_p, tf.global_variables()):
        if '1' in pp.name:
            print(pp.name)
            print(p[0:5])
    #check fixed_parameters
    print('-'*50)
    print('fixed parmas')
    w_list, b_list, a_list = sess.run(
            [model.fixed_w, model.fixed_b, model.fixed_a],
            feed_dict = {x: dataset.test.images}
            )
    w_clip, b_clip = sess.run([model.clip_list_w, model.clip_list_b])

    delta_w = w_clip[1]*2 / (2**2-2)
    delta_b = b_clip[1]*2 / (2**8-2)
    delta_a = (6.0-0) / (2**8-1)
    print('w_1')
    print('shape : ', w_list[1].shape)
    print('fake quant : ', w_list[1][0:5])
    print('clip value : ', w_clip[1])
    print('delta : ',delta_w) 
    print(' /delta : ', w_list[1][0:5] / delta_w)


    print('b_1')
    print('shape : ', b_list[1].shape)
    print('fake quant : ', b_list[1][0:5])
    print('clip value : ', b_clip[1])
    print('delta : ',delta_b) 
    print(' / delta : ', b_list[1][0:5] / delta_b)

    print('activation_1')
    print('shape : ', a_list[1].shape)
    print('fake quant : ', a_list[1][0:5])
    print('clip value : 6.0')
    print('delta : ',delta_a) 
    print(' / delta : ', a_list[1][0:5] / delta_a)
    

    exit()
    saver.save(sess, save_path = './results/model_fixed')

        




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
