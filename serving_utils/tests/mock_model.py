import tensorflow as tf


class MockModel:

    def __init__(self, maxlen=10):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x_place, self.y_place, \
                self.loss, self.train_op = self._build_graph(maxlen=maxlen)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(
            tf.variables_initializer(
                var_list=self.graph.get_collection("variables"),
            ),
        )

    def _build_graph(self, maxlen=10):
        x_place = tf.placeholder(
            shape=[None, maxlen], dtype=tf.float32, name='x')
        y_place = tf.placeholder(
            shape=[None, maxlen], dtype=tf.float32, name='y')\

        weights = tf.get_variable(
            name="weights",
            shape=[x_place.shape[-1], maxlen],
            dtype=tf.float32,
            trainable=True,
        )
        bias = tf.get_variable(
            name="bias",
            shape=[maxlen],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True,
        )
        prediction = tf.nn.relu(x_place @ weights + bias)
        prediction = tf.identity(prediction, name='pred')

        loss = tf.reduce_mean(tf.norm(y_place - prediction, axis=1))

        tvars = tf.trainable_variables()

        optimizer = tf.train.AdamOptimizer(1e-3)
        gradvar = optimizer.compute_gradients(
            loss=loss,
            var_list=tvars,
        )
        clipped_grad = [
            (tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradvar]
        train_op = optimizer.apply_gradients(clipped_grad)
        return x_place, y_place, loss, train_op

    def fit(self, x, y, epochs):
        for _ in range(epochs):
            train_loss, _ = self.sess.run(
                [self.loss, self.train_op],
                feed_dict={
                    self.x_place: x,
                    self.y_place: y,
                },
            )

    def evaluate(self, x, y):
        return self.sess.run(self.loss.name, feed_dict={self.x_place.name: x, self.y_place.name: y})

    def define_signature(self):
        signature_def_map = {
            'evaluate':
                tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs={
                        'x': self.x_place,
                        'y': self.y_place,
                    },
                    outputs={
                        'loss': self.loss,
                    },
                ),
        }
        return signature_def_map
