import tensorflow as tf
from serving_utils import Saver


graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    a = tf.Variable(1)
    b = tf.Variable(2)
    c = a + 2 * b
    sess.run(tf.global_variables_initializer())

    saver = Saver(
        session=sess,
        output_dir='./.fake-models/test_model',
        signature_def_map={
            'predict': tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={'a': a, 'b': b},
                outputs={'c': c},
            )
        }
    )

    saver.save()
