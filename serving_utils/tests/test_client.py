import asyncio as aio
import random

import pytest
from unittest import mock
from asynctest import CoroutineMock
from unittest.mock import patch
from grpclib.exceptions import GRPCError
from grpclib.const import Status
from ..client import Client

import numpy as np
from serving_utils import PredictInput


req_data = [
    PredictInput(name='a', value=np.int16(2)),
    PredictInput(name='b', value=np.int16(3)),
]
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


@pytest.mark.asyncio
async def test_load_balancing():

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

    def clear_created():
        created_grpc_channels.clear()
        created_grpclib_channels.clear()
        created_stubs.clear()
        created_async_stubs.clear()

    def assert_n_connections(n):
        assert_n_unique_mocks(created_grpc_channels, 'target', n)
        assert_n_unique_mocks(created_grpclib_channels, 'addr', n)
        assert_n_unique_mocks(created_stubs, 'channel', n)
        assert_n_unique_mocks(created_async_stubs, 'channel', n)

    with patch('socket.gethostbyname_ex') as mock_gethostbyname_ex:
        with patch('serving_utils.client.Channel',
                   side_effect=create_a_fake_grpclib_channel), \
            patch('serving_utils.client.grpc.secure_channel',
                  side_effect=create_a_fake_grpc_channel), \
            patch('serving_utils.client.grpc.insecure_channel',
                  side_effect=create_a_fake_grpc_channel), \
            patch('serving_utils.client.prediction_service_grpc.PredictionServiceStub',
                  side_effect=create_a_fake_async_stub), \
            patch('serving_utils.client.prediction_service_pb2_grpc.PredictionServiceStub',
                  side_effect=create_a_fake_stub), \
            patch.object(Client,
                         '_check_address_health'):

            # Case: Host name resolves to 1 IP address
            mock_gethostbyname_ex.return_value = ('localhost', [], ['1.2.3.4'])

            c = Client(host='localhost', port=9999)
            assert_n_connections(1)

            created_async_stubs[0].Predict.assert_not_awaited()

            await client_async_predict(c)

            created_async_stubs[0].Predict.assert_awaited()

            # Case: Host name resolves to 2 IP addresses
            clear_created()
            mock_gethostbyname_ex.return_value = ('localhost', [], ['1.2.3.4', '5.6.7.8'])

            c = Client(host='localhost', port=9999)

            assert_n_connections(2)

            await client_async_predict(c)
            await client_async_predict(c)

            created_async_stubs[0].Predict.assert_awaited()
            created_async_stubs[1].Predict.assert_awaited()

            # Case: Host name resolves to 3 IP address
            clear_created()
            mock_gethostbyname_ex.return_value = (
                'localhost', [], ['1.2.3.4', '5.6.7.8', '9.10.11.12'])

            c = Client(host='localhost', port=9999)

            assert_n_connections(3)

            await client_async_predict(c)
            await client_async_predict(c)
            await client_async_predict(c)

            created_async_stubs[0].Predict.assert_awaited()
            created_async_stubs[1].Predict.assert_awaited()
            created_async_stubs[2].Predict.assert_awaited()

            client_predict(c)
            await client_async_predict(c)

            assert created_stubs[0].Predict.call_count == 1
            assert created_async_stubs[1].Predict.await_count == 2


@pytest.mark.asyncio
async def test_server_reset_handling():

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

    def clear_created():
        created_grpc_channels.clear()
        created_grpclib_channels.clear()
        created_stubs.clear()
        created_async_stubs.clear()

    def mock_host_reset(mock_gethostbyname_ex, new_addr_list):
        mock_gethostbyname_ex.side_effect = lambda _: ('localhost', [], new_addr_list.copy())
        for stub in created_stubs:
            stub.Predict.side_effect = GRPCError(Status.UNAVAILABLE)
        for stub in created_async_stubs:
            stub.Predict.side_effect = GRPCError(Status.UNAVAILABLE)

    def assert_n_connections(c, n):
        assert len(c._pool) == n

    with patch('socket.gethostbyname_ex') as mock_gethostbyname_ex:
        with patch('serving_utils.client.Channel',
                   side_effect=create_a_fake_grpclib_channel), \
            patch('serving_utils.client.grpc.secure_channel',
                  side_effect=create_a_fake_grpc_channel), \
            patch('serving_utils.client.grpc.insecure_channel',
                  side_effect=create_a_fake_grpc_channel), \
            patch('serving_utils.client.prediction_service_grpc.PredictionServiceStub',
                  side_effect=create_a_fake_async_stub), \
            patch('serving_utils.client.prediction_service_pb2_grpc.PredictionServiceStub',
                  side_effect=create_a_fake_stub), \
            patch.object(Client,
                         '_check_address_health'):

            # Case: Host name resolves to 0 IP addresses
            mock_host_reset(mock_gethostbyname_ex, [])

            c = Client(host='localhost', port=9999)
            assert_n_connections(c, 0)

            # Case: Host reset, resolves to 1 IP addresses
            mock_host_reset(mock_gethostbyname_ex, ['1.2.3.4'])

            await client_async_predict(c)
            assert_n_connections(c, 1)

            # Case: Host reset, resolves to 1 different IP address
            mock_host_reset(mock_gethostbyname_ex, ['5.6.7.8'])

            await client_async_predict(c)
            assert_n_connections(c, 1)

            # Case: Host reset, resolves to 2 different IP address
            mock_host_reset(mock_gethostbyname_ex, ['10.10.10.10', '11.11.11.11'])

            await client_async_predict(c)
            assert_n_connections(c, 2)

            await client_async_predict(c)
            await client_async_predict(c)
            await client_async_predict(c)
            await client_async_predict(c)
            await client_async_predict(c)

            conns = dict(c._pool._list)
            assert conns['10.10.10.10'].async_stub.Predict.await_count == 3
            assert conns['11.11.11.11'].async_stub.Predict.await_count == 3

            # Case: Host reset, 1 extra IP added
            mock_host_reset(mock_gethostbyname_ex, ['10.10.10.10', '11.11.11.11', '12.12.12.12'])
            await client_async_predict(c)

            conns = dict(c._pool._list)
            assert conns['10.10.10.10'].async_stub.Predict.await_count == 3
            assert conns['11.11.11.11'].async_stub.Predict.await_count == 3
            assert conns['12.12.12.12'].async_stub.Predict.await_count == 1

            # Case: Host reset, 1 IP removed and 2 new IPs added
            mock_host_reset(mock_gethostbyname_ex,
                ['10.10.10.10', '11.11.11.11', '13.13.13.13', '14.14.14.14'])
            await client_async_predict(c)
            await client_async_predict(c)

            conns = dict(c._pool._list)
            assert conns['10.10.10.10'].async_stub.Predict.await_count == 3
            assert conns['11.11.11.11'].async_stub.Predict.await_count == 3
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
                    mock_host_reset(
                        mock_gethostbyname_ex,
                        [str(random.randint(0, 1000)) for _ in range(random.randint(1, 5))],
                    )
                    await aio.sleep(random.random() + 0.5)

            t = aio.ensure_future(aio.gather(
                loop_calling_async_predict(c),
                loop_host_reset(c, mock_gethostbyname_ex),
            ))
            await aio.sleep(3)
            t.cancel()
            try:
                await t
            except aio.CancelledError:
                pass
