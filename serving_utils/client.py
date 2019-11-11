from functools import partial
import itertools
import socket
from typing import List
import time

import asyncio
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import grpc
from grpclib.client import Channel
import tensorflow as tf

from .protos import predict_pb2, prediction_service_pb2_grpc, list_models_pb2, list_models_pb2_grpc
from .protos import prediction_service_grpc


def copy_message(src, dst):
    """
    Copy the contents of a src proto message to a destination proto message via string serialization
    :param src: Source proto
    :param dst: Destination proto
    :return:
    """
    dst.ParseFromString(src.SerializeToString())
    return dst


PredictInput = namedtuple('PredictInput', ['name', 'value'])


class Client:

    CHECK_ADDRESS_HEALTH_TIMES = 4
    TIMEOUT_SECONDS = 5

    def __init__(
            self,
            host: str,
            port: int,
            pem: str = None,
            channel_options: dict = None,
            loop: asyncio.AbstractEventLoop = None,
            executor: ThreadPoolExecutor = None,
            standalone_pool_for_streaming: bool = False,
        ):
        """Constructor.

        Args:
            addr (str) : address of your serving
            pem: credentials of grpc
            channel+options: An optional list of key-value pairs (channel args in gRPC runtime)
            loop: asyncio event loop
            executor: a thread pool, or None to use the default pool of the loop
            standalone_pool_for_streaming: create a new thread pool (with 1 thread)
                for each streaming method
        """
        self.addr = f"{host}:{port}"
        if channel_options is None:
            channel_options = {}
        if pem is None:
            make_sync_channel = grpc.insecure_channel
        else:
            creds = grpc.ssl_channel_credentials(pem)
            make_sync_channel = partial(grpc.secure_channel, credentials=creds)

        if loop is None:
            loop = asyncio.get_event_loop()

        _, _, addrlist = socket.gethostbyname_ex(host)
        channels = []
        async_channels = []
        stubs = []
        async_stubs = []
        for a in addrlist:
            channels.append(make_sync_channel(
                f"{a}:{port}",
                options=channel_options,
            ))
            stubs.append(prediction_service_pb2_grpc.PredictionServiceStub(channels[-1]))

            async_channels.append(Channel(a, port, loop=loop))
            async_stubs.append(prediction_service_grpc.PredictionServiceStub(async_channels[-1]))

        self._stubs = stubs
        self._async_stubs = async_stubs
        self._channels = channels
        self._async_channels = async_channels
        self._round_robin_cycler = itertools.cycle(range(len(addrlist)))

        self._check_address_health()

    def _check_address_health(self):
        for _ in range(self.CHECK_ADDRESS_HEALTH_TIMES):
            time.sleep(1)
            req = predict_pb2.PredictRequest()
            req.model_spec.name = 'intentionally_missing_model'
            try:
                self._stub.Predict(req, self.TIMEOUT_SECONDS)
            except Exception as e:
                _code = e._state.code
                if _code == grpc.StatusCode.UNAVAILABLE:
                    raise e
                elif _code == grpc.StatusCode.NOT_FOUND:
                    break
                else:
                    raise Exception(e)
        else:
            raise ConnectionError(f"Connection Timeout for addr: {self.addr}")

    @staticmethod
    def _predict_request(
            data,
            model_name,
            output_names=None,
            model_signature_name=None,
        ):
        req = predict_pb2.PredictRequest()
        req.model_spec.name = model_name
        if model_signature_name is not None:
            req.model_spec.signature_name = model_signature_name

        for datum in data:
            copy_message(tf.make_tensor_proto(datum.value), req.inputs[datum.name])
        if output_names is not None:
            for output_name in output_names:
                req.output_filter.append(output_name)
        return req

    @staticmethod
    def parse_predict_response(response):
        results = {}
        for key in response.outputs:
            tensor_proto = response.outputs[key]
            nd_array = tf.contrib.util.make_ndarray(tensor_proto)
            results[key] = nd_array
        return results

    def list_models(self):
        stub = list_models_pb2_grpc.ListModelsStub(self._channel)
        response = stub.ListModels(list_models_pb2.ListModelsRequest())
        return response.models

    def get_round_robin_stub(self, is_async_stub=False):
        i = next(self._round_robin_cycler)
        if is_async_stub:
            return self._async_stubs[i]
        else:
            return self._stubs[i]

    def predict(
            self,
            data: List[PredictInput],
            output_names: List[str] = None,
            model_name: str = 'default',
            model_signature_name: str = None,
        ):
        request = self._predict_request(
            data=data,
            output_names=output_names,
            model_name=model_name,
            model_signature_name=model_signature_name,
        )
        stub = self.get_round_robin_stub(is_async_stub=False)
        response = stub.Predict(request)
        return self.parse_predict_response(response)

    async def async_predict(
            self,
            data: List[PredictInput],
            output_names: List[str] = None,
            model_name: str = 'default',
            model_signature_name: str = None,
        ):
        request = self._predict_request(
            data=data,
            output_names=output_names,
            model_name=model_name,
            model_signature_name=model_signature_name,
        )
        stub = self.get_round_robin_stub(is_async_stub=True)
        response = await stub.Predict(request)
        return self.parse_predict_response(response)
