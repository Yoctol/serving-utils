import collections


class RoundRobinMap(collections.abc.MutableMapping):

    def __init__(self):
        self._container = collections.OrderedDict()

    def __delitem__(self, k):
        del self._container[k]

    def __getitem__(self, k):
        v = self._container[k]
        self._container.move_to_end(k)
        return v

    def __iter__(self):
        if not self._container:
            raise StopIteration

        first_key = next(iter(self._container))
        v = self._container[first_key]
        self._container.move_to_end(first_key)
        yield (first_key, v)

    def __len__(self):
        return len(self._container)

    def __setitem__(self, k, v):
        self._container[k] = v
        self._container.move_to_end(k, last=False)

    def keys(self):
        return self._container.keys()
