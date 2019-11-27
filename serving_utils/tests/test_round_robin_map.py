from .. import round_robin_map


def test_RoundRobinMap():
    pool = round_robin_map.RoundRobinMap()

    assert len(pool) == 0
    assert pool.keys() == set()

    pool['a'] = 'abc'
    assert len(pool) == 1
    assert pool.keys() == {'a'}

    assert next(iter(pool)) == ('a', 'abc')
    assert next(iter(pool)) == ('a', 'abc')
    assert next(iter(pool)) == ('a', 'abc')

    pool['b'] = 'bbc'
    assert len(pool) == 2
    assert pool.keys() == {'a', 'b'}

    assert next(iter(pool)) == ('b', 'bbc')
    assert next(iter(pool)) == ('a', 'abc')
    assert next(iter(pool)) == ('b', 'bbc')
    assert next(iter(pool)) == ('a', 'abc')

    pool['c'] = 'ccc'
    assert pool.keys() == {'a', 'b', 'c'}

    assert next(iter(pool)) == ('c', 'ccc')
    assert next(iter(pool)) == ('b', 'bbc')
    assert next(iter(pool)) == ('a', 'abc')

    del pool['c']
    assert pool.keys() == {'a', 'b'}

    assert next(iter(pool)) == ('b', 'bbc')
    assert next(iter(pool)) == ('a', 'abc')

    assert next(iter(pool)) == ('b', 'bbc')
    pool['b'] = 1
    assert next(iter(pool)) == ('b', 1)
    pool['b'] = 1
    assert next(iter(pool)) == ('b', 1)
    pool['b'] = 1
    assert next(iter(pool)) == ('b', 1)

    assert next(iter(pool)) == ('a', 'abc')
    pool['b']
    assert next(iter(pool)) == ('a', 'abc')
    pool['b']
    assert next(iter(pool)) == ('a', 'abc')
