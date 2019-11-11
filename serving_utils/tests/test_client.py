import pytest
from unittest import mock
from asynctest import CoroutineMock
from unittest.mock import patch
from ..client import Client

import numpy as np
from serving_utils import PredictInput


@pytest.mark.asyncio
async def test_load_balancing():

    created_channels = []
    created_async_stubs = []
    created_stubs = []

    def create_a_fake_channel(*_, **__):
        m = mock.MagicMock()
        created_channels.append(m)
        return m

    def create_a_fake_async_stub(_):
        m = mock.MagicMock()
        m.Predict = CoroutineMock()
        created_async_stubs.append(m)
        return m

    def create_a_fake_stub(_):
        m = mock.MagicMock()
        created_stubs.append(m)
        return m

    with patch('socket.gethostbyname_ex', return_value=('hostname', [], ['1.2.3.4'])):
        with patch('serving_utils.client.Channel',
                   side_effect=create_a_fake_channel), \
            patch('serving_utils.client.prediction_service_grpc.PredictionServiceStub',
                  side_effect=create_a_fake_async_stub), \
            patch('serving_utils.client.prediction_service_pb2_grpc.PredictionServiceStub',
                  side_effect=create_a_fake_stub), \
            patch.object(Client,
                         '_check_address_health'):

            c = Client('127.0.0.1:9999')
            req_data = [
                PredictInput(name='a', value=np.int16(2)),
                PredictInput(name='b', value=np.int16(3)),
            ]
            output_names = ['c']
            model_name = 'test_model'

            await c.async_predict(
                data=req_data,
                output_names=output_names,
                model_name=model_name,
                model_signature_name='test',
            )

            assert len(created_channels) == 1
            assert len(created_async_stubs) == 1
            created_async_stubs[0].Predict.assert_awaited()

    created_channels.clear()
    created_async_stubs.clear()
    created_stubs.clear()

    with patch('socket.gethostbyname_ex', return_value=('hostname', [], ['1.2.3.4', '5.6.7.8'])):
        with patch('serving_utils.client.Channel',
                   side_effect=create_a_fake_channel), \
            patch('serving_utils.client.prediction_service_grpc.PredictionServiceStub',
                  side_effect=create_a_fake_async_stub), \
            patch('serving_utils.client.prediction_service_pb2_grpc.PredictionServiceStub',
                  side_effect=create_a_fake_stub), \
            patch.object(Client,
                         '_check_address_health'):

            c = Client('127.0.0.1:9999')
            req_data = [
                PredictInput(name='a', value=np.int16(2)),
                PredictInput(name='b', value=np.int16(3)),
            ]
            output_names = ['c']
            model_name = 'test_model'

            await c.async_predict(
                data=req_data,
                output_names=output_names,
                model_name=model_name,
                model_signature_name='test',
            )
            await c.async_predict(
                data=req_data,
                output_names=output_names,
                model_name=model_name,
                model_signature_name='test',
            )

            assert len(created_channels) == 2
            assert len(created_async_stubs) == 2
            created_async_stubs[0].Predict.assert_awaited()
            created_async_stubs[1].Predict.assert_awaited()

    created_channels.clear()
    created_async_stubs.clear()
    created_stubs.clear()

    with patch('socket.gethostbyname_ex',
               return_value=('hostname', [], ['1.2.3.4', '5.6.7.8', '9.10.11.12'])):
        with patch('serving_utils.client.Channel',
                   side_effect=create_a_fake_channel), \
            patch('serving_utils.client.prediction_service_grpc.PredictionServiceStub',
                  side_effect=create_a_fake_async_stub), \
            patch('serving_utils.client.prediction_service_pb2_grpc.PredictionServiceStub',
                  side_effect=create_a_fake_stub), \
            patch.object(Client,
                         '_check_address_health'):

            c = Client('127.0.0.1:9999')
            req_data = [
                PredictInput(name='a', value=np.int16(2)),
                PredictInput(name='b', value=np.int16(3)),
            ]
            output_names = ['c']
            model_name = 'test_model'

            await c.async_predict(
                data=req_data,
                output_names=output_names,
                model_name=model_name,
                model_signature_name='test',
            )
            await c.async_predict(
                data=req_data,
                output_names=output_names,
                model_name=model_name,
                model_signature_name='test',
            )
            await c.async_predict(
                data=req_data,
                output_names=output_names,
                model_name=model_name,
                model_signature_name='test',
            )

            assert len(created_channels) == 3
            assert len(created_async_stubs) == 3
            created_async_stubs[0].Predict.assert_awaited()
            created_async_stubs[1].Predict.assert_awaited()
            created_async_stubs[2].Predict.assert_awaited()

            c.predict(
                data=req_data,
                output_names=output_names,
                model_name=model_name,
                model_signature_name='test',
            )
            await c.async_predict(
                data=req_data,
                output_names=output_names,
                model_name=model_name,
                model_signature_name='test',
            )

            assert created_stubs[0].Predict.call_count == 1
            assert created_async_stubs[1].Predict.await_count == 2
