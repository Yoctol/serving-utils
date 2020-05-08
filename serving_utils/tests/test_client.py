import asyncio as aio
import random

import pytest
from unittest import mock
from asynctest import CoroutineMock
from unittest.mock import patch
import grpc
import grpc._channel
import grpclib
import numpy as np

from ..client import Client, RetryFailed, Connection


req_data = {
    'a': np.int16(2),
    'b': np.int16(3),
}
output_names = ['c']
model_name = 'test_model'


def client_predict(c):
    c.predict(
        data=req_data,
        output_names=output_names,
        model_name=model_name,
        model_signature_name='test',
    )


async def client_async_predict(c):
    await c.async_predict(
        data=req_data,
        output_names=output_names,
        model_name=model_name,
        model_signature_name='test',
    )


def assert_n_unique_mocks(mocks, attr, n):
    assert len(mocks) == n
    s = set(getattr(m, attr) for m in mocks)
    print(s)
    assert len(s) == n


def setup_function(f):

    created_grpc_channels = []
    created_grpclib_channels = []
    created_stubs = []
    created_async_stubs = []

    def assert_n_unique_mocks(mocks, attr, n):
        assert len(mocks) == n
        s = set(getattr(m, attr) for m in mocks)
        print(s)
        assert len(s) == n

    def create_a_fake_grpclib_channel(addr, port, loop):
        m = mock.MagicMock(name=f"{addr}:{port}")
        m.addr = addr
        created_grpclib_channels.append(m)
        return m

    def create_a_fake_grpc_channel(target, *_, **__):
        m = mock.MagicMock(name=target)
        m.target = target
        created_grpc_channels.append(m)
        return m

    def assert_n_connections(n):
        assert_n_unique_mocks(created_grpc_channels, 'target', n)
        assert_n_unique_mocks(created_grpclib_channels, 'addr', n)
        assert_n_unique_mocks(created_stubs, 'channel', n)
        assert_n_unique_mocks(created_async_stubs, 'channel', n)

    def clear_created():
        created_grpc_channels.clear()
        created_grpclib_channels.clear()
        created_stubs.clear()
        created_async_stubs.clear()

    def create_a_fake_async_stub(mock_channel):
        m = mock.MagicMock(name=f"async stub channel={mock_channel}")
        m.channel = mock_channel
        m.Predict = CoroutineMock()
        created_async_stubs.append(m)
        return m

    def create_a_fake_stub(mock_channel):
        m = mock.MagicMock(name=f"stub channel={mock_channel}")
        m.channel = mock_channel
        created_stubs.append(m)
        return m

    def hostname_resolution_change(mock_gethostbyname_ex, new_addr_list):
        mock_gethostbyname_ex.side_effect = lambda _: ('localhost', [], new_addr_list.copy())
        for stub in created_stubs:
            addr = stub.channel.target.split(':')[0]
            if addr not in new_addr_list:
                stub.Predict.side_effect = create_grpc_error("UNAVAILABLE", "", sync=True)
            else:
                stub.Predict.side_effect = None
        for stub in created_async_stubs:
            addr = stub.channel.addr
            if addr not in new_addr_list:
                stub.Predict.side_effect = create_grpc_error("UNAVAILABLE", "", sync=False)
            else:
                stub.Predict.side_effect = None

    f.mock_gethostbyname_ex = patch('socket.gethostbyname_ex').start()
    patch('serving_utils.client.Channel',
          side_effect=create_a_fake_grpclib_channel).start()
    patch('serving_utils.client.grpc.secure_channel',
          side_effect=create_a_fake_grpc_channel).start()
    f.patch_insecure_channel = patch(
        'serving_utils.client.grpc.insecure_channel',
        side_effect=create_a_fake_grpc_channel,
    ).start()
    patch('serving_utils.client.prediction_service_grpc.PredictionServiceStub',
          side_effect=create_a_fake_async_stub).start()
    patch('serving_utils.client.prediction_service_pb2_grpc.PredictionServiceStub',
          side_effect=create_a_fake_stub).start()

    f.created_grpc_channels = created_grpc_channels
    f.created_grpclib_channels = created_grpclib_channels
    f.created_stubs = created_stubs
    f.created_async_stubs = created_async_stubs

    f.assert_n_connections = assert_n_connections
    f.clear_created = clear_created
    f.hostname_resolution_change = hostname_resolution_change


def teardown_function(f):
    patch.stopall()


@pytest.mark.asyncio
async def test_load_balancing():
    t = test_load_balancing

    # Case: Host name resolves to 1 IP address
    t.mock_gethostbyname_ex.return_value = ('localhost', [], ['1.2.3.4'])

    c = Client(host='localhost', port=9999, n_trys=1)
    t.assert_n_connections(1)

    t.created_async_stubs[0].Predict.assert_not_awaited()

    await client_async_predict(c)

    t.created_async_stubs[0].Predict.assert_awaited()

    # Case: Host name resolves to 2 IP addresses
    t.clear_created()
    t.mock_gethostbyname_ex.return_value = ('localhost', [], ['1.2.3.4', '5.6.7.8'])

    c = Client(host='localhost', port=9999)

    t.assert_n_connections(2)

    await client_async_predict(c)
    await client_async_predict(c)

    t.created_async_stubs[0].Predict.assert_awaited()
    t.created_async_stubs[1].Predict.assert_awaited()

    # Case: Host name resolves to 3 IP address
    t.clear_created()
    t.mock_gethostbyname_ex.return_value = (
        'localhost', [], ['1.2.3.4', '5.6.7.8', '9.10.11.12'])

    c = Client(host='localhost', port=9999)

    t.assert_n_connections(3)

    await client_async_predict(c)
    await client_async_predict(c)
    await client_async_predict(c)
    client_predict(c)
    client_predict(c)
    client_predict(c)

    for stub in t.created_async_stubs + t.created_stubs:
        assert stub.Predict.call_count == 1


def create_grpc_error(status, message, *, sync):
    if sync:
        status = grpc.StatusCode[status]

        rpc_state = grpc._channel._RPCState(
            grpc._channel._UNARY_UNARY_INITIAL_DUE, None, None, None, None)
        rpc_state.code = status
        rpc_state.details = message
        return grpc._channel._Rendezvous(rpc_state, None, None, None)
    else:
        # Create grpc.RpcError
        # https://github.com/grpc/grpc/blob/c1d176528fd8da9dd4066d16554bcd216d29033f/src/python/grpcio/grpc/_channel.py#L592
        status = grpclib.const.Status[status]
        return grpclib.exceptions.GRPCError(status, message=message)


@pytest.mark.asyncio
async def test_model_not_found_error_passes_through_async_predict():

    t = test_model_not_found_error_passes_through_async_predict
    t.mock_gethostbyname_ex.return_value = ('localhost', [], ['1.2.3.4'])

    # pyserving will send this kind of error when there is no such model
    expected_exception = create_grpc_error("NOT_FOUND", "Model XXX not found", sync=False)

    def server_fails_to_Predict_because_model_doesnt_exist(request):
        raise expected_exception

    mock_logger = mock.Mock()
    c = Client(host='localhost', port=9999, n_trys=1, logger=mock_logger)
    for stub in t.created_async_stubs:
        stub.Predict.side_effect = server_fails_to_Predict_because_model_doesnt_exist

    with pytest.raises(grpclib.exceptions.GRPCError) as exc_info:
        await client_async_predict(c)

    assert exc_info.value.status == grpclib.const.Status.NOT_FOUND
    assert exc_info.value.message == "Model XXX not found"


@pytest.mark.asyncio
async def test_asyncio_cancel_during_async_predict():
    t = test_asyncio_cancel_during_async_predict
    t.mock_gethostbyname_ex.return_value = ('localhost', [], ['1.2.3.4'])

    mock_logger = mock.Mock()
    c = Client(host='localhost', port=9999, n_trys=1, logger=mock_logger)
    for stub in t.created_async_stubs:
        stub.Predict.side_effect = aio.CancelledError()

    with pytest.raises(aio.CancelledError):
        await client_async_predict(c)


def test_RetryFailed():
    e1 = Exception("abc")
    e2 = Exception("xyz")
    e3 = create_grpc_error("UNKNOWN", "wtf", sync=False)
    e4 = create_grpc_error("ABORTED", "wtf", sync=True)

    retry_error = RetryFailed(message="my message", errors=[e1, e2, e3, e4])

    assert 'my message' in str(retry_error)
    assert 'abc' in str(retry_error)
    assert 'xyz' in str(retry_error)
    assert "ABORTED" in str(retry_error)
    assert "UNKNOWN" in str(retry_error)


def test_Connection_with_channel_options():
    t = test_Connection_with_channel_options
    channel_options = [('max_receive_body', 128 * 1000 * 1000)]
    Connection('127.0.0.1', 8500, channel_options=channel_options)
    t.patch_insecure_channel.assert_called_once_with(
        '127.0.0.1:8500',
        options=channel_options,
    )


@pytest.mark.asyncio
async def test_model_not_found_error_passes_through_sync_predict():

    t = test_model_not_found_error_passes_through_sync_predict
    t.mock_gethostbyname_ex.return_value = ('localhost', [], ['1.2.3.4'])

    # pyserving will send this kind of error when there is no such model
    expected_exception = create_grpc_error("NOT_FOUND", "Model XXX not found", sync=True)
    assert isinstance(expected_exception, grpc.RpcError)

    def server_fails_to_Predict_because_model_doesnt_exist(request):
        raise expected_exception

    mock_logger = mock.Mock()
    c = Client(host='localhost', port=9999, n_trys=1, logger=mock_logger)
    for stub in t.created_stubs:
        stub.Predict.side_effect = server_fails_to_Predict_because_model_doesnt_exist

    with pytest.raises(grpc.RpcError) as exc_info:
        client_predict(c)

    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND
    assert exc_info.value.details() == "Model XXX not found"


@pytest.mark.asyncio
async def test_retrying():
    t = test_retrying

    t.mock_gethostbyname_ex.return_value = ('localhost', [], ['1.2.3.4'])

    exceptions_for_Predict = [
        Exception(f"{i}. I'm sorry. I can't do that") for i in range(5)
    ]

    for n_trys in range(1, 5):

        t.clear_created()
        mock_logger = mock.Mock()
        c = Client(host='localhost', port=9999, n_trys=n_trys, logger=mock_logger)
        for stub in t.created_stubs + t.created_async_stubs:
            stub.Predict.side_effect = exceptions_for_Predict

        with patch.object(
                Client,
                '_setup_connections',
                wraps=c._setup_connections,
            ) as mock_setup_connections:

            with pytest.raises(RetryFailed) as exc_info:
                await client_async_predict(c)
            e = exc_info.value
            assert e.errors == exceptions_for_Predict[:n_trys]

        assert mock_setup_connections.call_count >= n_trys
        mock_logger.exception.assert_has_calls(
            [mock.call(e) for e in exceptions_for_Predict[:n_trys]])

        total_calls = 0
        for stub in t.created_async_stubs:
            total_calls += stub.Predict.call_count
        assert total_calls == n_trys

        mock_logger.reset_mock()
        with patch.object(
                Client,
                '_setup_connections',
                wraps=c._setup_connections,
            ) as mock_setup_connections:

            with pytest.raises(RetryFailed):
                client_predict(c)
            e = exc_info.value
            assert e.errors == exceptions_for_Predict[:n_trys]

        assert mock_setup_connections.call_count >= n_trys
        mock_logger.exception.assert_has_calls(
            [mock.call(e) for e in exceptions_for_Predict[:n_trys]])

        total_calls = 0
        for stub in t.created_stubs:
            total_calls += stub.Predict.call_count
        assert total_calls == n_trys


@pytest.mark.asyncio
async def test_handle_hostname_resolution_change():
    t = test_handle_hostname_resolution_change

    def assert_poolsize(c, n):
        assert len(c._pool) == n

    # Case: Host name resolves to 0 IP addresses
    t.hostname_resolution_change(t.mock_gethostbyname_ex, [])

    c = Client(host='localhost', port=9999, n_trys=1)
    assert_poolsize(c, 0)

    # Case: Host reset, resolves to 1 IP addresses
    t.hostname_resolution_change(t.mock_gethostbyname_ex, ['1.2.3.4'])

    await client_async_predict(c)
    assert_poolsize(c, 1)

    # Case: Host reset, resolves to 1 different IP address
    t.hostname_resolution_change(t.mock_gethostbyname_ex, ['5.6.7.8'])

    await client_async_predict(c)
    assert_poolsize(c, 1)

    # Case: Host reset, resolves to 2 different IP address
    t.hostname_resolution_change(t.mock_gethostbyname_ex, ['10.10.10.10', '11.11.11.11'])

    await client_async_predict(c)
    assert_poolsize(c, 2)

    await client_async_predict(c)
    await client_async_predict(c)
    await client_async_predict(c)
    await client_async_predict(c)
    await client_async_predict(c)

    conns = c._pool._container
    assert conns['10.10.10.10'].async_stub.Predict.await_count == 3
    assert conns['11.11.11.11'].async_stub.Predict.await_count == 3

    # Case: Host reset, 1 extra IP added
    t.hostname_resolution_change(
        t.mock_gethostbyname_ex,
        ['10.10.10.10', '11.11.11.11', '12.12.12.12'],
    )
    await client_async_predict(c)
    await client_async_predict(c)
    await client_async_predict(c)

    conns = c._pool._container
    assert conns['10.10.10.10'].async_stub.Predict.await_count == 4
    assert conns['11.11.11.11'].async_stub.Predict.await_count == 4
    assert conns['12.12.12.12'].async_stub.Predict.await_count == 1

    # Case: Host reset, 1 IP removed and 2 new IPs added
    t.hostname_resolution_change(
        t.mock_gethostbyname_ex,
        ['10.10.10.10', '11.11.11.11', '13.13.13.13', '14.14.14.14'],
    )
    await client_async_predict(c)
    await client_async_predict(c)
    await client_async_predict(c)
    await client_async_predict(c)

    conns = c._pool._container
    assert conns['10.10.10.10'].async_stub.Predict.await_count == 5
    assert conns['11.11.11.11'].async_stub.Predict.await_count == 5
    assert conns['13.13.13.13'].async_stub.Predict.await_count == 1
    assert conns['14.14.14.14'].async_stub.Predict.await_count == 1

    # Fuzz test
    # Randomly reset host a number of times
    # Testing that old conns are never used, if they are used an exception
    # will be raised
    async def loop_calling_async_predict(client):
        while True:
            n = random.randint(1, 5)
            await aio.gather(*[client_async_predict(client) for _ in range(n)])
            await aio.sleep(random.random())

    async def loop_host_reset(client, mock_gethostbyname_ex):
        while True:
            t.hostname_resolution_change(
                mock_gethostbyname_ex,
                [str(random.randint(0, 1000)) for _ in range(random.randint(1, 5))],
            )
            await aio.sleep(random.random() + 0.5)

    task = aio.ensure_future(aio.gather(
        loop_calling_async_predict(c),
        loop_host_reset(c, t.mock_gethostbyname_ex),
    ))
    await aio.sleep(3)
    task.cancel()
    try:
        await task
    except aio.CancelledError:
        pass
