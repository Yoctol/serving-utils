import pytest
from unittest import mock
from asynctest import CoroutineMock
from unittest.mock import patch
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

    def create_a_fake_grpclib_channel(addr, port, loop):
        m = mock.MagicMock(name=f"{addr}:{port}")
        created_grpclib_channels.append(m)
        return m

    def create_a_fake_grpc_channel(target, *_, **__):
        m = mock.MagicMock(name=target)
        created_grpc_channels.append(m)
        return m

    def create_a_fake_async_stub(*args, **kwargs):
        m = mock.MagicMock(name=f"async stub {args} {kwargs}")
        m.Predict = CoroutineMock()
        created_async_stubs.append(m)
        return m

    def create_a_fake_stub(*args, **kwargs):
        m = mock.MagicMock(name=f"stub {args} {kwargs}")
        created_stubs.append(m)
        return m

    def clear_created():
        created_grpc_channels.clear()
        created_grpclib_channels.clear()
        created_stubs.clear()
        created_async_stubs.clear()

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

            mock_gethostbyname_ex.return_value=('localhost', [], ['1.2.3.4'])

            c = Client('localhost:9999')
            assert len(created_grpc_channels) == 1
            assert len(created_grpclib_channels) == 1
            assert len(created_stubs) == 1
            assert len(created_async_stubs) == 1

            created_async_stubs[0].Predict.assert_not_awaited()
           
            await client_async_predict(c) 

            created_async_stubs[0].Predict.assert_awaited()


            clear_created()
            mock_gethostbyname_ex.return_value = ('localhost', [], ['1.2.3.4', '5.6.7.8'])
            
            c = Client('localhost:9999')
            assert len(created_grpc_channels) == 2
            assert len(created_grpclib_channels) == 2
            assert len(created_stubs) == 2
            assert len(created_async_stubs) == 2

            await client_async_predict(c)
            await client_async_predict(c)

            created_async_stubs[0].Predict.assert_awaited()
            created_async_stubs[1].Predict.assert_awaited()

            clear_created()
            mock_gethostbyname_ex.return_value = (
                'localhost', [], ['1.2.3.4', '5.6.7.8', '9.10.11.12'])

            c = Client('localhost:9999')
            assert len(created_grpc_channels) == 3
            assert len(created_grpclib_channels) == 3
            assert len(created_stubs) == 3
            assert len(created_async_stubs) == 3

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
