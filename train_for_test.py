import tensorflow as tf
from serving_utils import Saver


graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    a = tf.Variable(1, dtype=tf.int16)
    b = tf.Variable(2, dtype=tf.int16)
    c = a + 2 * b
    sess.run(tf.global_variables_initializer())

    saver = Saver(
        session=sess,
        output_dir='./.fake-models/test_model',
        signature_def_map={
            'test': tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={'a': a, 'b': b},
                outputs={'c': c},
            )
        }
    )

    saver.save()
