import tensorflow as tf



class simple_mlp(object):
    def __init__(self, n_layer, hidden_dim):
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        
        self.w_list = []
        self.b_list = []

        input_dim = 784
        output_dim = 10
        for i in range(n_layer):
            w = tf.get_variable(
                    name = 'w_'+str(i), 
                    shape = [input_dim, hidden_dim],
                    initializer = tf.glorot_uniform_initializer()
                    )
            b = tf.get_variable(
                    name = 'b_'+str(i),
                    shape = [hidden_dim],
                    initializer = tf.zeros_initializer()
                    )
            self.w_list.append(w)
            self.b_list.append(b)
            input_dim = hidden_dim

        w = tf.get_variable(
                name = 'w_'+str(n_layer),
                shape = [hidden_dim, output_dim],
                initializer = tf.glorot_uniform_initializer()
                )
        b = tf.get_variable(
                name = 'b_'+str(n_layer),
                shape = [output_dim],
                initializer = tf.zeros_initializer()
                )
        self.w_list.append(w)
        self.b_list.append(b)



    def __call__(self, inputs, n_w = 8, n_b = 8, n_a = 8):
        self.fixed_w = []
        self.fixed_b = []
        self.fixed_a = []
        self.clip_list_w = []
        self.clip_list_b = []

        for i in range(self.n_layer):
            w = self.w_list[i]
            w_mean, w_var = tf.nn.moments(w, axes = [0,1])
            w_clip = 1.2*tf.sqrt(w_var)
            #w_clip = 0.6*tf.sqrt(w_var) * (2**(_N_W-1))
            self.clip_list_w.append(w_clip)
            w = tf.quantization.fake_quant_with_min_max_vars(
                    inputs = w,
                    min = -1.0*w_clip,
                    max = w_clip,
                    num_bits = n_w,
                    narrow_range = True
                    )
            self.fixed_w.append(w)


            b = self.b_list[i]
            b_mean, b_var = tf.nn.moments(b, axes = [0])
            b_clip = 1.2*tf.sqrt(b_var)
            self.clip_list_b.append(b_clip)
            #b_clip = 0.6*tf.sqrt(b_var) * (2**(_N_B-1))
            b = tf.quantization.fake_quant_with_min_max_vars(
                    inputs = b,
                    min = -1.0*b_clip,
                    max = b_clip,
                    num_bits = n_b,
                    narrow_range = True
                    )
            self.fixed_b.append(b)
            inputs = tf.nn.xw_plus_b(inputs, w, b)
            inputs = tf.nn.relu(inputs)
            inputs = tf.quantization.fake_quant_with_min_max_args(
                    inputs = inputs,
                    min = 0,
                    max = 6.0,
                    num_bits = n_a,
                    narrow_range = False
                    )
            self.fixed_a.append(inputs)
        
        logit = tf.nn.xw_plus_b(inputs, self.w_list[self.n_layer], self.b_list[self.n_layer])

        return logit




