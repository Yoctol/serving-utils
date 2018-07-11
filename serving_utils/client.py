import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Mapping

from aiogrpc import Channel
import grpc
import numpy as np
import tensorflow as tf

from .protos import predict_pb2, prediction_service_pb2_grpc


class Client:

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
        self._async_channel = Channel(
            self._channel,
            loop=loop,
            executor=executor,
            standalone_pool_for_streaming=standalone_pool_for_streaming,
        )

        self._stub = prediction_service_pb2_grpc.PredictionServiceStub(self._channel)
        self._async_stub = prediction_service_pb2_grpc.PredictionServiceStub(self._async_channel)

    @staticmethod
    def _predict_request(data, output_names=None, model_name=None):
        if model_name is None:
            raise ValueError("model_name must be provided.")
        req = predict_pb2.PredictRequest()
        req.model_spec.name = model_name
        for datum in data:
            req.inputs[datum['name']].CopyFrom(
                tf.contrib.util.make_tensor_proto(datum['value']))
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

    def predict(
            self,
            data: List[Mapping[str, np.ndarray]],
            output_names: List[str] = None,
            model_name: str = None,
        ):
        request = self._predict_request(
            data=data,
            output_names=output_names,
            model_name=model_name,
        )
        response = self._stub.Predict(request)
        return self.parse_predict_response(response)

    async def async_predict(
            self,
            data: List[Mapping[str, np.ndarray]],
            output_names: List[str] = None,
            model_name: str = None,
        ):
        request = self._predict_request(
            data=data,
            output_names=output_names,
            model_name=model_name,
        )
        response = await self._async_stub.Predict(request)
        return self.parse_predict_response(response)
