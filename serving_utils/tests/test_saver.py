from os.path import abspath, dirname, join, isdir
from glob import glob
from shutil import rmtree

import tensorflow as tf
import numpy as np

from .mock_model import MockModel
from ..saver import Saver

ROOT_DIR = dirname(abspath(__file__))


class SaverTestCase(tf.test.TestCase):

    def setUp(self):
        self.model = MockModel(maxlen=10)
        self.output_dir = join(ROOT_DIR, 'model_test/')
        self.saver = Saver(
            session=self.model.sess,
            output_dir=self.output_dir,
            signature_def_map=self.model.define_signature(),
        )
        self.x_test = np.random.rand(5, 10).astype('float32')
        self.y_test = np.random.rand(5, 10).astype('float32')
        self.output_filenames = set(
            [
                join(self.output_dir, '0', filename) for filename in [
                    'saved_model.pb',
                    'variables',
                    'variables/variables.data-00000-of-00001',
                    'variables/variables.index',
                ]
            ],
        )

    def tearDown(self):
        if isdir(self.output_dir):
            rmtree(self.output_dir)

    def test_save_correct_files(self):
        self.saver.save()
        self.assertEqual(
            self.output_filenames,
            set(
                glob(self.output_dir + '/*/*') +  # noqa: W504
                glob(self.output_dir + '/*/*/*'),
            ),
        )

    def test_save_will_not_change_model(self):
        old_loss = self.model.evaluate(self.x_test, self.y_test)
        self.saver.save()
        new_loss = self.load_n_evaluate()
        self.assertEqual(old_loss, new_loss)

    def load_n_evaluate(self):
        with tf.Session(graph=tf.Graph()) as sess:
            meta_graph_def = tf.saved_model.loader.load(
                sess=sess,
                tags=[tf.saved_model.tag_constants.SERVING],
                export_dir=join(self.output_dir, '0'),
            )
            evaluate_graph = meta_graph_def.signature_def['evaluate']
            loss = sess.run(
                evaluate_graph.outputs['loss'].name,
                feed_dict={
                    evaluate_graph.inputs['x'].name: self.x_test,
                    evaluate_graph.inputs['y'].name: self.y_test,
                },
            )
        return loss

    def test_freeze_graph_has_session_update(self):
        old_sess = self.saver.session
        self.saver.freeze_graph()
        new_sess = self.saver.session
        self.assertNotEqual(old_sess, new_sess)

    def test_freeze_graph_will_not_change_loss(self):
        old_loss = self.model.evaluate(self.x_test, self.y_test)
        self.saver.freeze_graph()
        self.model.sess = self.saver.session
        new_loss = self.model.evaluate(self.x_test, self.y_test)
        self.assertEqual(old_loss, new_loss)

    def test_freeze_n_save(self):
        self.saver.freeze_graph()
        self.saver.save()
