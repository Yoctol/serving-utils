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
            addr: str,
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
        self.addr = addr
        if channel_options is None:
            channel_options = {}
        if pem is None:
            self._channel = grpc.insecure_channel(
                addr,
                options=channel_options,
            )
        else:
            creds = grpc.ssl_channel_credentials(pem)
            self._channel = grpc.secure_channel(
                addr,
                credentials=creds,
                options=channel_options,
            )

        if loop is None:
            loop = asyncio.get_event_loop()

        split_addr = addr.split(':')
        host = split_addr[0]
        port = split_addr[1]
        # TODO: better addr parsing, secure channel
        self._async_channel = Channel(host, port, loop=loop)

        self._stub = prediction_service_pb2_grpc.PredictionServiceStub(self._channel)
        self._async_stub = prediction_service_grpc.PredictionServiceStub(self._async_channel)
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
        response = self._stub.Predict(request)
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
        response = await self._async_stub.Predict(request)
        return self.parse_predict_response(response)
