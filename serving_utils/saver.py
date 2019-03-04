from typing import Dict
import pathlib

import tensorflow as tf
from tensorflow.python.framework import graph_util


class Saver:

    def __init__(
            self,
            session: tf.Session,
            output_dir: str,
            signature_def_map: Dict[str, tf.saved_model.signature_def_utils.predict_signature_def],
        ) -> None:
        self.session = session
        self.signature_def_map = signature_def_map
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_next_version(path: pathlib.PosixPath) -> pathlib.PosixPath:
        candidate_paths = path.glob('**/*')
        dirs = [x.name for x in candidate_paths if x.is_dir()]
        versions = [int(dir_name) for dir_name in dirs if dir_name.isdigit()]
        new_version = max(versions) + 1 if versions else 0
        return path / str(new_version)

    def save(self, **kwargs) -> str:
        output_version_dir = str(self._get_next_version(self.output_dir))
        with self.session.graph.as_default():
            builder = tf.saved_model.builder.SavedModelBuilder(
                export_dir=output_version_dir,
            )
            builder.add_meta_graph_and_variables(
                sess=self.session,
                tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map=self.signature_def_map,
            )
        builder.save()
        return output_version_dir

    def freeze_graph(self):
        """Freeze Graph and update session"""

        output_op_names = []
        for signature_value in self.signature_def_map.values():
            for _, tensor_info in signature_value.outputs.items():
                tensor_name = tensor_info.name
                if ':' in tensor_name:
                    op_name = tensor_name.split(':')[0]
                else:
                    op_name = tensor_name
                output_op_names.append(op_name)

        frozen_graph_def = graph_util.convert_variables_to_constants(
            sess=self.session,
            input_graph_def=self.session.graph_def,
            output_node_names=output_op_names,
        )

        new_sess = tf.Session(graph=tf.Graph())
        with new_sess.graph.as_default():
            tf.import_graph_def(frozen_graph_def, name="")

        # update session
        self.session = new_sess
