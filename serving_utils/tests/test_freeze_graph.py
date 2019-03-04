import pytest
import numpy as np

from ..freeze_graph import freeze_graph, create_session_from_graphdef
from .mock_model import MockModel


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


def test_create_session_from_graphdef(model):
    old_graph_def = model.sess.graph_def
    new_sess = create_session_from_graphdef(model.sess.graph_def)
    assert old_graph_def == new_sess.graph_def


def test_freeze_graph_will_not_change_loss(x_test, y_test, model):
    old_loss = model.evaluate(x_test, y_test)
    graph_def = freeze_graph(model.sess, [model.loss.op.name])

    # model get new sess created from frozen graph def
    new_sess = create_session_from_graphdef(graph_def)
    model.sess = new_sess
    new_loss = model.evaluate(x_test, y_test)

    assert old_loss == new_loss


def test_op_not_in_graph(model):
    with pytest.raises(KeyError):
        freeze_graph(model.sess, ['not_in_graph'])
