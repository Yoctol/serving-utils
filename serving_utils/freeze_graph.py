import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph(session, output_op_names):
    """Freeze graph

    Freeze graph accroding to input session and
    output operation names

    ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
        python/tools/freeze_graph.py

    Args:
        session: an instance of tf.Session
        output_op_names (strs): a list of operation names

    Return:
        a tf.GraphDef object

    """
    operations_in_graph(graph_def=session.graph_def, op_names=output_op_names)
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess=session,
        input_graph_def=session.graph_def,
        output_node_names=output_op_names,  # node name == op name
    )
    return frozen_graph_def


def operations_in_graph(graph_def, op_names):
    """
    Check whether all operations are in graph or not

    Args:
        graph_def (tf.GraphDef): a tf.GraphDef object
        op_names (strs): a list of operation names

    Return:
        KeyError or None

    """
    all_node_names = [node.name for node in graph_def.node]
    for op_name in op_names:
        if op_name not in all_node_names:
            raise KeyError(f"{op_name} is not in this graph.")


def create_session_from_graphdef(graph_def):
    """
    Create new session from given tf.GraphDef object

    Arg:
       graph_def (tf.GraphDef): a tf.GraphDef object

    Return:
       session (tf.Session): a new session with given graph_def

    """
    new_sess = tf.Session(graph=tf.Graph())
    with new_sess.graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return new_sess
