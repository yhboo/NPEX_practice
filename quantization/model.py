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



    def __call__(self, inputs):
        for i in range(self.n_layer):
            inputs = tf.nn.xw_plus_b(inputs, self.w_list[i], self.b_list[i])
            inputs = tf.nn.relu(inputs)
        
        logit = tf.nn.xw_plus_b(inputs, self.w_list[self.n_layer], self.b_list[self.n_layer])

        return logit




