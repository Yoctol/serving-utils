import pytest
from os.path import abspath, dirname, join, isdir
from glob import glob
from shutil import rmtree

import tensorflow as tf
import numpy as np

from .mock_model import MockModel
from ..saver import Saver

ROOT_DIR = dirname(abspath(__file__))


@pytest.fixture
def output_dir():
    return join(ROOT_DIR, 'model_test/')


@pytest.fixture
def x_test():
    return np.random.rand(5, 10).astype('float32')


@pytest.fixture
def y_test(x_test):
    batch_size, dims = x_test.shape
    return np.random.rand(batch_size, dims).astype('float32')


@pytest.fixture
def model(x_test):
    return MockModel(maxlen=x_test.shape[1])


@pytest.fixture
def saver(model, output_dir):
    yield Saver(
        session=model.sess,
        output_dir=output_dir,
        signature_def_map=model.define_signature(),
        freeze=False,
    )
    if isdir(output_dir):
        rmtree(output_dir)


@pytest.fixture
def saver_with_freeze(model, output_dir):
    yield Saver(
        session=model.sess,
        output_dir=output_dir,
        signature_def_map=model.define_signature(),
    )
    if isdir(output_dir):
        rmtree(output_dir)


def test_save_correct_files(saver, output_dir):
    saver.save()
    expected_output_filenames = set(
        [
            join(output_dir, '0', filename) for filename in [
                'saved_model.pb',
                'variables',
                'variables/variables.data-00000-of-00001',
                'variables/variables.index',
            ]
        ],
    )
    real_output_filenames = set(
        glob(output_dir + '/*/*') +  # noqa: W504
        glob(output_dir + '/*/*/*'),
    )
    assert expected_output_filenames == real_output_filenames


def test_save_will_not_change_model(x_test, y_test, model, saver, output_dir):
    old_loss = model.evaluate(x_test, y_test)
    saver.save()
    new_loss = load_n_evaluate(model.sess, output_dir, x_test, y_test)
    assert old_loss == new_loss


def load_n_evaluate(sess, output_dir, x_test, y_test):
    with tf.Session(graph=tf.Graph()) as sess:
        meta_graph_def = tf.saved_model.loader.load(
            sess=sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            export_dir=join(output_dir, '0'),
        )
        evaluate_graph = meta_graph_def.signature_def['evaluate']
        loss = sess.run(
            evaluate_graph.outputs['loss'].name,
            feed_dict={
                evaluate_graph.inputs['x'].name: x_test,
                evaluate_graph.inputs['y'].name: y_test,
            },
        )
    return loss


def test_frozen_save_correct_files(saver_with_freeze, output_dir):
    saver_with_freeze.save()
    expected_output_filenames = set(
        [
            join(output_dir, '0', filename) for filename in [
                'saved_model.pb',
                'variables',
            ]
        ],
    )
    real_output_filenames = set(
        glob(output_dir + '/*/*') +  # noqa: W504
        glob(output_dir + '/*/*/*'),
    )
    assert expected_output_filenames == real_output_filenames

    # def test_freeze_graph_has_session_update(self):
    #     old_sess = saver.session
    #     saver.freeze_graph()
    #     new_sess = saver.session
    #     assertNotEqual(old_sess, new_sess)

    # def test_freeze_graph_will_not_change_loss(self):
    #     old_loss = model.evaluate(x_test, y_test)
    #     saver.freeze_graph()
    #     model.sess = saver.session
    #     new_loss = model.evaluate(x_test, y_test)
    #     assertEqual(old_loss, new_loss)

    # def test_freeze_n_save(self):
    #     saver.freeze_graph()
    #     saver.save()
