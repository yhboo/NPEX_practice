import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model import simple_mlp



_N_W = 2
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
    logit = model(x)
    
    #define loss and optimizers
    loss = tf.losses.sparse_softmax_cross_entropy(labels = y, logits = logit) 
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(tf.argmax(logit, axis=-1)), y)))


    ################################changes#################################
    #print graph nodes before create quantize graph
    print('-'*50)
    print('before quantization')
    for p in tf.global_variables():
        print(p.name)
    #print([n.name for n in tf.get_default_graph().as_graph_def().node if 'w_1' in n.name])
    saver = tf.train.Saver()

    #create quantization nodes
    g = tf.get_default_graph()
    #tf.contrib.quantize.create_eval_graph(input_graph = g)
    
    tf.contrib.quantize.experimental_create_eval_graph(
            input_graph = g,
            weight_bits = _N_W,
            activation_bits = _N_A
            )
    
    print('-'*50)
    print('after quantization')
    for p in tf.global_variables():
        print(p.name)
    #exit()
    #print([n.name for n in tf.get_default_graph().as_graph_def().node if 'act_quant' in n.name])
    w1_fixed = g.get_tensor_by_name('xw_plus_b_1/weights_quant/FakeQuantWithMinMaxVars:0')
    a1_fixed = g.get_tensor_by_name('xw_plus_b_1/act_quant/FakeQuantWithMinMaxVars:0')
    #######################################################################
    
    

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
    np.set_printoptions(precision=4, suppress = True)
    print('-'*50)
    print('variables')
    for p, pp in zip(all_p, tf.global_variables()):
        if '1' in pp.name:
            print(pp.name)
            if 'min' in pp.name or 'max' in pp.name:
                print(p)
            else:
                print(p[0:5])
    
    #exit()
    w1f, a1f = sess.run(
            [w1_fixed, a1_fixed],
            feed_dict = {x:dataset.test.images}
            )
    w_clip = (6.0 + 6.0) / (2**_N_W-2) 
    a_clip = (6.0 - 0.0) / (2**_N_A-1) 
    print('w_1')
    print(w1f[0:5])
    print('delta : ', w_clip)
    print('/delta : ', w1f[0:5] / w_clip)
    print('a_1')
    print(a1f[0:5])
    print('delta : ', a_clip)
    print('/delta : ', a1f[0:5] / a_clip)
    exit()
    saver.save(sess, save_path = './results/model_fixed_created')


        




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
