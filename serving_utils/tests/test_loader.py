from unittest import TestCase
import shutil

import tensorflow as tf
import numpy as np

from ..saver import Saver
from ..loader import Loader


class LoaderTestCase(TestCase):

    def setUp(self):
        graph = tf.Graph()
        self.sess = tf.Session(graph=graph)
        with graph.as_default():
            self.a_place = tf.placeholder(shape=(1, 10), dtype=tf.float32, name='input_a')
            self.b_place = tf.placeholder(shape=(1, 10), dtype=tf.float32, name='input_b')
            self.z_place = tf.placeholder(shape=(1, 10), dtype=tf.float32, name='input_z')
            self.another_b_place = tf.placeholder(shape=(1, 10), dtype=tf.float32, name='other_b')
            self.c_tensor = 3 * self.a_place
            self.d_tensor = self.c_tensor + self.b_place ** 2
            self.e_tensor = 5 * self.another_b_place
            self.w_tensor = 7 * self.z_place
        self.signature_def_map = {
            'predict_c': tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={'a': self.a_place},
                outputs={'c': self.c_tensor},
            ),
            'predict_d': tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={'a': self.a_place, 'b': self.b_place},
                outputs={'d': self.d_tensor},
            ),
            'predict_e': tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={'b': self.another_b_place},
                outputs={'e': self.e_tensor},
            ),
            'predict_w': tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={'z': self.z_place},
                outputs={'w': self.w_tensor},
            ),
        }
        self.serving_path = '/tmp/model_for_serving_model_loader'

        Saver(
            session=self.sess,
            output_dir=self.serving_path,
            signature_def_map=self.signature_def_map,
        ).save()

    def tearDown(self):
        self.sess.close()
        shutil.rmtree(self.serving_path)

    def test_default_tags_correct(self):
        sml = Loader(self.serving_path)
        assert sml._tags == [tf.saved_model.tag_constants.SERVING]

    def test_load_empty(self):
        new_sess = tf.Session(graph=tf.Graph())
        sml = Loader(self.serving_path)
        sml.load(new_sess)

        with new_sess:
            input_a_val = np.random.rand(1, 10).astype(np.float32)
            c_val = new_sess.run(
                self.c_tensor.name,
                feed_dict={self.a_place.name: input_a_val},
            )
            np.testing.assert_array_almost_equal(
                c_val,
                input_a_val * 3,
            )

            input_b_val = np.random.rand(1, 10).astype(np.float32)
            d_val = new_sess.run(
                self.d_tensor.name,
                feed_dict={
                    self.a_place.name: input_a_val,
                    self.b_place.name: input_b_val,
                },
            )
            np.testing.assert_array_almost_equal(
                d_val,
                input_a_val * 3 + input_b_val ** 2,
            )

    def test_load_wo_signature_key(self):
        new_sess = tf.Session(graph=tf.Graph())
        sml = Loader(self.serving_path)
        with new_sess.graph.as_default():
            a = tf.placeholder(shape=(1, 10), dtype=tf.float32, name='new_input_a')
            double_a = tf.multiply(a, 2, name="double_a")
        input_name_map = {'a': double_a}
        sml.load(new_sess, input_name_map)

        with new_sess:
            input_a_val = np.random.rand(1, 10).astype(np.float32)
            c_val = new_sess.run(
                self.c_tensor.name,
                feed_dict={a: input_a_val},
            )
            np.testing.assert_array_almost_equal(
                c_val,
                input_a_val * 3 * 2,
            )

            input_b_val = np.random.rand(1, 10).astype(np.float32)
            d_val = new_sess.run(
                self.d_tensor.name,
                feed_dict={
                    a: input_a_val,
                    self.b_place.name: input_b_val,
                },
            )
            np.testing.assert_array_almost_equal(
                d_val,
                input_a_val * 3 * 2 + input_b_val ** 2,
            )

    def test_load_wo_signature_key_but_input_name_not_in_first_signature(self):
        new_sess = tf.Session(graph=tf.Graph())
        sml = Loader(self.serving_path)
        with new_sess.graph.as_default():
            z = tf.placeholder(shape=(1, 10), dtype=tf.float32, name='new_input_z')
            double_z = tf.multiply(z, 2, name="double_z")
        input_name_map = {'z': double_z}
        sml.load(new_sess, input_name_map)

        with new_sess:
            input_z_val = np.random.rand(1, 10).astype(np.float32)
            w_val = new_sess.run(
                self.w_tensor.name,
                feed_dict={
                    z: input_z_val,
                },
            )
            np.testing.assert_array_almost_equal(
                w_val,
                input_z_val * 2 * 7,
            )

    def test_load_raise_if_input_name_is_not_consistent(self):
        new_sess = tf.Session(graph=tf.Graph())
        sml = Loader(self.serving_path)
        with new_sess.graph.as_default():
            b = tf.placeholder(shape=(1, 10), dtype=tf.float32, name='new_input_b')
        input_name_map = {'b': b}

        with self.assertRaises(ValueError):
            sml.load(new_sess, input_name_map)

    def test_load_raise_if_input_name_is_not_exist(self):
        new_sess = tf.Session(graph=tf.Graph())
        sml = Loader(self.serving_path)
        with new_sess.graph.as_default():
            b = tf.placeholder(shape=(1, 10), dtype=tf.float32, name='new_input_b')
        input_name_map = {'g': b}

        with self.assertRaises(KeyError):
            sml.load(new_sess, input_name_map)

    def test_load_raise_if_input_name_is_not_in_specified_signature(self):
        new_sess = tf.Session(graph=tf.Graph())
        sml = Loader(self.serving_path)
        with new_sess.graph.as_default():
            b = tf.placeholder(shape=(1, 10), dtype=tf.float32, name='new_input_b')
        input_name_map = {'a': b}

        with self.assertRaises(KeyError):
            sml.load(new_sess, input_name_map, signature_key='predict_e')

    def test_load_w_signature_key(self):
        new_sess = tf.Session(graph=tf.Graph())
        sml = Loader(self.serving_path)
        with new_sess.graph.as_default():
            b = tf.placeholder(shape=(1, 10), dtype=tf.float32, name='new_input_b')
            tripple_b = b * 3

        input_name_map = {'b': tripple_b}
        sml.load(new_sess, input_name_map, signature_key='predict_e')

        with new_sess:
            input_b_val = np.random.rand(1, 10).astype(np.float32)
            e_val = new_sess.run(
                self.e_tensor.name,
                feed_dict={b: input_b_val},
            )
            np.testing.assert_array_almost_equal(
                e_val,
                input_b_val * 3 * 5,
            )

    def test_load_with_version(self):
        graph = tf.Graph()
        another_sess = tf.Session(graph=graph)
        with graph.as_default():
            a_place = tf.placeholder(shape=(1, 10), dtype=tf.float32, name='input_a')
            c_tensor = 2 * a_place
        another_signature_def_map = {
            'predict_c': tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={'a': a_place},
                outputs={'c': c_tensor},
            ),
        }

        # version 1
        Saver(
            session=another_sess,
            output_dir=self.serving_path,
            signature_def_map=another_signature_def_map,
        ).save()

        # version 2
        Saver(
            session=self.sess,
            output_dir=self.serving_path,
            signature_def_map=self.signature_def_map,
        ).save()

        sml = Loader(self.serving_path, version=1)

        new_sess = tf.Session(graph=tf.Graph())
        sml.load(new_sess)

        with new_sess:
            input_a_val = np.random.rand(1, 10).astype(np.float32)
            c_val = new_sess.run(
                c_tensor.name,
                feed_dict={a_place.name: input_a_val},
            )

            np.testing.assert_array_almost_equal(
                c_val,
                input_a_val * 2,
            )
