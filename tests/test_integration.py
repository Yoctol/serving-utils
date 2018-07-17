# integrations tests
# for unit tests, go to the `tests` folder of each submodules
import time
import pytest

import numpy as np
from serving_utils import Client, PredictInput


@pytest.mark.integration
@pytest.mark.asyncio
async def test_client():
    test_serving_ports = [
        8501,
        8502,
        8503,
        8504,
    ]

    while True:
        for serving_port in test_serving_ports:
            time.sleep(1)
            try:
                client = Client(
                    addr=f"localhost:{serving_port}",
                )
                client.predict(None, output_names='wrong_model', model_signature_name='test')
                break

            except Exception:
                continue
        else:
            break
    # test client predict correct result
    # fake model is generated from `train_for_test.py`
    clients = []
    for serving_port in test_serving_ports:
        clients.append(Client(
            addr=f"localhost:{serving_port}",
        ))

    req_data = [
        PredictInput(name='a', value=np.int16(2)),
        PredictInput(name='b', value=np.int16(3)),
    ]
    output_names = ['c']
    model_name = 'test_model'
    expected_output = {'c': 8}  # c = a + 2 * b
    for client in clients:
        actual_output = client.predict(
            data=req_data,
            output_names=output_names,
            model_name=model_name,
            model_signature_name='test',
        )

        assert actual_output == expected_output

        actual_output = await client.async_predict(
            data=req_data,
            output_names=output_names,
            model_name=model_name,
            model_signature_name='test',
        )

        assert actual_output == expected_output
