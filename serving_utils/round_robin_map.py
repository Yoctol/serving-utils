import collections


class RoundRobinMap(collections.abc.MutableMapping):

    def __init__(self):
        self._list = []

    def __delitem__(self, k):
        for i, (key, _) in enumerate(self._list):
            if key == k:
                self._list.pop(i)
                return
        else:
            return

    def __getitem__(self, k):

        for i, (key, val) in enumerate(self._list):
            if key == k:
                self._list.pop(i)
                self._list.append((key, val))
                return val
        else:
            raise KeyError(k)

    def __iter__(self):
        if not self._list:
            raise StopIteration

        k, v = self._list.pop(0)
        self._list.append((k, v))
        yield (k, v)

    def __len__(self):
        return len(self._list)

    def __setitem__(self, k, v):
        for i, (key, _) in enumerate(self._list):
            if key == k:
                self._list.pop(i)
                break
        self._list.insert(0, (k, v))

    def keys(self):
        return {k for (k, _) in self._list}
