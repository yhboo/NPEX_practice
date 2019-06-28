import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model_fixed import simple_mlp


_N_W = 2
_N_B = 8
_N_A = 8


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

    optimizer = tf.train.GradientDescentOptimizer(0.1)

    train_op = optimizer.minimize(loss)

    saver = tf.train.Saver()
    ###### end graph build ######


    ### call graph through tf.Session() ###
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './results/model')

    ###training example###
    n_epoch = 10
    n_iter = int(n_epoch*50000/100)
    for _ in range(n_iter):
        batch_x, batch_y = dataset.train.next_batch(100)
        sess.run(train_op,
                feed_dict = {x: batch_x, y : batch_y}
                )
    

    ### test trained model ###
    test_loss, test_acc = sess.run(
            [loss, acc],
            feed_dict = {x: dataset.test.images, y: dataset.test.labels}
            )

    print('test loss : %.4f, acc : %.4f'%(test_loss, test_acc))
    '''    
    all_p = sess.run(tf.global_variables())

    #check parameters
    print('-'*50)
    print('float parmas')
    for p, pp in zip(all_p, tf.global_variables()):
        if '1' in pp.name:
            print(pp.name)
            print(p)

    #check fixed_parameters
    print('-'*50)
    print('fixed parmas')
    w_list, b_list, a_list = sess.run(
            [model.fixed_w, model.fixed_b, model.fixed_a],
            feed_dict = {x: dataset.test.images}
            )
    print('w_1')
    print('shape : ', w_list[1].shape)
    print('fake quant : ', w_list[1])
    print(' /delta : ', w_list[1] / (0.25 / (2**(2-1)-1)))


    print('b_1')
    print('shape : ', b_list[1].shape)
    print('fake quant : ', b_list[1])
    print(' / delta : ', b_list[1] / (0.5 / (2**(4-1)-1)))

    print('activation_1')
    print('shape : ', a_list[1].shape)
    print('fake quant : ', a_list[1])
    print(' / delta : ', a_list[1] / (1.0 / (2**8-1)))
    
    '''
    exit()
    saver.save(sess, save_path = './results/model_fixed')


        




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
