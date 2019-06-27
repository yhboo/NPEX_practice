import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model import simple_mlp




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
    logit = model(x)
    
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

    ###training example###
    n_epoch = 50
    n_iter = int(50*50000/100)
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
    
    all_p = sess.run(tf.global_variables())

    #check parameters
    for p, pp in zip(all_p, tf.global_variables()):
        print(pp.name, ' : ', p.shape)
        if '1' in pp.name:
            print(p)

    saver.save(sess, save_path = './results/model')


        




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
