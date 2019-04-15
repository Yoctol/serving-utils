import os
import pathlib
from typing import List, Mapping

import tensorflow as tf
from tensorflow.python.saved_model.loader_impl import SavedModelLoader


def get_latest_version(path):
    path = pathlib.Path(path)
    candidate_paths = path.glob('**/*')
    dirs = [x.name for x in candidate_paths if x.is_dir()]
    versions = [int(dir_name) for dir_name in dirs if dir_name.isdigit()]
    return max(versions)


class Loader:
    """An adapter of tensorflow.python.saved_model.loader_impl.SavedModelLoader"""

    def __init__(
            self,
            path: str,
            tags: List[str] = None,
            version: int = None,
        ):
        self._path = path
        self._tags = tags or [tf.saved_model.tag_constants.SERVING]
        if version is None:
            version = get_latest_version(self._path)

        serving_path_with_version = os.path.join(self._path, str(version))
        self._sml = SavedModelLoader(serving_path_with_version)
        self._meta_graph = self._sml.get_meta_graph_def_from_tags(self._tags)

    @property
    def meta_graph(self):
        return self._meta_graph

    @property
    def signature_def(self):
        return self._meta_graph.signature_def

    def load(
            self,
            sess,
            input_name_map: Mapping[str, tf.Tensor] = None,  # signature_input_name => new_tensor
            signature_key: str = None,  # if None, will check all key of same name
        ):
        """Load the saved model to session, and connect to current graph.

        >>> sess = tf.Session()
        >>> new_x = tf.placeholder()
        >>> sml = ServingModelLoader('/path/to/serving')
        >>> input_name_map = {'x': new_x}
        >>> sml.load(sess, input_name_map)
        """
        if input_name_map is None:
            input_name_map = {}

        self._check_input_name_map_valid_with_signature_key(signature_key, input_name_map)

        input_map = {}
        for name, tensor in input_name_map.items():
            if signature_key is not None:
                old_tensor_name = self.signature_def[signature_key].inputs[name].name
            else:
                for signature in self.signature_def.values():
                    if name in signature.inputs:
                        old_tensor_name = signature.inputs[name].name
                        break
            input_map[old_tensor_name] = tensor

        self._sml.load(
            sess,
            tags=self._tags,
            input_map=input_map,
        )

    def _check_input_name_map_valid_with_signature_key(self, signature_key, input_name_map):
        if signature_key is None:
            for name in input_name_map:
                self._check_name_consistent_among_signatures(name)
        else:
            for name in input_name_map:
                self._check_name_in_signature_key(name, signature_key)

    def _check_name_consistent_among_signatures(self, input_name):
        all_tensor_name = []
        for signature in self.signature_def.values():
            if input_name in signature.inputs:
                all_tensor_name.append(signature.inputs[input_name].name)

        if len(set(all_tensor_name)) == 0:  # means the name does not exist in this signature_def
            msg = f"{input_name} does not exist in any signatures"
            raise KeyError(msg)

        if len(set(all_tensor_name)) != 1:  # if tensor names are unique
            msg = f"{input_name} does not point to same tensor in all signatures"
            raise ValueError(msg)

    def _check_name_in_signature_key(self, input_name, signature_key):
        if input_name not in self.signature_def[signature_key].inputs:
            msg = f"{input_name} does not exist in signature {signature_key}"
            raise KeyError(msg)
