from typing import Dict
import logging
import os

from mkdir_p import mkdir_p
import tensorflow as tf

LOGGER = logging.getLogger(__name__)


class Saver:

    def __init__(
            self,
            session: tf.Session,
            output_dir: str,
            signature_def_map: Dict[str, tf.saved_model.signature_def_utils.predict_signature_def],
            logger: object = LOGGER,
        ) -> None:
        self.session = session
        self.signature_def_map = signature_def_map
        self.output_dir = output_dir
        mkdir_p(self.output_dir)
        self.logger = logger

    def _check_is_version(self, dir_name: str) -> bool:
        try:
            int(dir_name)
        except ValueError:
            return False
        return True

    def _get_next_version(self, path: str) -> str:
        dirs = os.listdir(path)
        max_version = -1
        for dir_name in dirs:
            if self._check_is_version(dir_name) and max_version < int(dir_name):
                max_version = int(dir_name)
        return os.path.join(path, str(max_version + 1))

    def save(
            self,
            legacy_init_op: tf.group = None,
            **kwargs
        ) -> None:
        output_version_dir = self._get_next_version(self.output_dir)

        self.logger.info(f"Saving model to {output_version_dir}")

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
