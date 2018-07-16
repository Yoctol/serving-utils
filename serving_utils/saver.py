from typing import Dict
import pathlib

import tensorflow as tf


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

    def _check_is_version(self, dir_name: str) -> bool:
        try:
            int(dir_name)
        except ValueError:
            return False
        return True

    def _get_next_version(
            self,
            path: pathlib.PosixPath,
        ) -> pathlib.PosixPath:
        candidate_paths = path.glob('**/*')
        dirs = [x.name for x in candidate_paths if x.is_dir()]
        max_version = -1
        for dir_name in dirs:
            if self._check_is_version(dir_name) and max_version < int(dir_name):
                max_version = int(dir_name)
        return path / str(max_version + 1)

    def save(
            self,
            legacy_init_op: tf.group = None,
            **kwargs
        ) -> str:
        output_version_dir = str(self._get_next_version(self.output_dir))
        with self.session.graph.as_default():
            builder = tf.saved_model.builder.SavedModelBuilder(
                export_dir=output_version_dir,
            )
            builder.add_meta_graph_and_variables(
                sess=self.session,
                tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map=self.signature_def_map,
                legacy_init_op=legacy_init_op,
            )
        builder.save()
        return output_version_dir
