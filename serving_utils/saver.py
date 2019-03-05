from typing import Dict
import pathlib
import tensorflow as tf

from .freeze_graph import freeze_graph, create_session_from_graphdef


class Saver:

    def __init__(
            self,
            session: tf.Session,
            output_dir: str,
            signature_def_map: Dict[str, tf.saved_model.signature_def_utils.predict_signature_def],
            freeze: bool = True,
        ) -> None:
        self.session = session
        self.signature_def_map = signature_def_map
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freeze = freeze

    def save(self, **kwargs) -> str:
        """Save model given session and signature

        Save frozen model if self.freeze is True.

        Return:
            output_version_dir

        """
        if self.freeze:
            output_op_names = self._get_op_names()
            frozen_graphdef = freeze_graph(self.session, output_op_names)
            session_to_be_saved = create_session_from_graphdef(frozen_graphdef)
        else:
            session_to_be_saved = self.session

        output_version_dir = str(self._get_next_version(self.output_dir))
        with session_to_be_saved.graph.as_default():
            builder = tf.saved_model.builder.SavedModelBuilder(
                export_dir=output_version_dir,
            )
            builder.add_meta_graph_and_variables(
                sess=session_to_be_saved,
                tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map=self.signature_def_map,
            )
        builder.save()

        if self.freeze:
            session_to_be_saved.close()

        return output_version_dir

    @staticmethod
    def _get_next_version(path: pathlib.PosixPath) -> pathlib.PosixPath:
        candidate_paths = path.glob('**/*')
        dirs = [x.name for x in candidate_paths if x.is_dir()]
        versions = [int(dir_name) for dir_name in dirs if dir_name.isdigit()]
        new_version = max(versions) + 1 if versions else 0
        return path / str(new_version)

    def _get_op_names(self):
        """
        Extract output operation names from signature

        Return:
            output_op_names (strs): a list of operation names
                for output recorded in signature.

        """
        output_op_names = []
        for signature_value in self.signature_def_map.values():
            for _, tensor_info in signature_value.outputs.items():
                tensor_name = tensor_info.name
                if ':' in tensor_name:
                    op_name = tensor_name.split(':')[0]
                else:
                    op_name = tensor_name
                output_op_names.append(op_name)
        return output_op_names
