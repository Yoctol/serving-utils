from os.path import abspath, dirname, join, isdir
from glob import glob
from shutil import rmtree

import tensorflow as tf
import numpy as np

from .mock_model import MockModel
from ..saver import Saver

ROOT_DIR = dirname(abspath(__file__))


class MockModelWithSave(MockModel):

    def __init__(self, maxlen=10):
        super(MockModelWithSave, self).__init__(maxlen=maxlen)
        self.output_dir = join(ROOT_DIR, 'model_test/')
        self.saver = Saver(
            session=self.sess,
            output_dir=self.output_dir,
            signature_def_map=self.define_signature(),
        )

    def save_tf_serve(self, legacy_init_op: tf.group = None):
        self.saver.save(
            legacy_init_op=legacy_init_op,
        )


class SaverTestCase(tf.test.TestCase):

    def setUp(self):
        self.model = MockModelWithSave(10)

    def tearDown(self):
        if isdir(self.model.output_dir):
            rmtree(self.model.output_dir)

    def test_fit_n_save_serving(self):
        self.model.save_tf_serve()
        self.assertEqual(
            set(
                [
                    join(
                        self.model.output_dir,
                        '0',
                        filename,
                    ) for filename in [
                        'saved_model.pb',
                        'variables',
                        'variables/variables.data-00000-of-00001',
                        'variables/variables.index',
                    ]
                ],
            ),
            set(
                glob(self.model.output_dir + '/*/*') +  # noqa:W504
                glob(self.model.output_dir + '/*/*/*'),
            ),
        )
        with tf.Session(graph=tf.Graph()) as sess:
            meta_graph_def = tf.saved_model.loader.load(
                sess=sess,
                tags=[tf.saved_model.tag_constants.SERVING],
                export_dir=join(
                    self.model.output_dir,
                    '0',
                ),
            )
            evaluate_graph = meta_graph_def.signature_def['evaluate']
            loss = sess.run(
                sess.graph.get_tensor_by_name(
                    evaluate_graph.outputs['loss'].name),
                feed_dict={
                    sess.graph.get_tensor_by_name(
                        evaluate_graph.inputs['x'].name):
                    np.random.rand(5, 10).astype('float32'),
                    sess.graph.get_tensor_by_name(
                        evaluate_graph.inputs['y'].name):
                    np.random.rand(5, 10).astype('float32'),
                },
            )
        self.assertEqual((), loss.shape)

    def test_fit_n_save_serving_checkpoint(self):
        x = np.random.rand(5, 10).astype('float32')
        y = 2 * x + 1
        self.model.fit(
            x=np.random.rand(5, 10).astype('float32'),
            y=y,
            epochs=2,
        )
