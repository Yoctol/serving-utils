# serving-utils

[![travis][travis-image]][travis-url]
[![pypi][pypi-image]][pypi-url]
![release][release-image]

[travis-image]: https://travis-ci.org/Yoctol/serving-utils.svg?branch=master
[travis-url]: https://travis-ci.org/Yoctol/serving-utils
[pypi-image]: https://img.shields.io/pypi/v/serving-utils.svg?style=flat
[pypi-url]: https://pypi.python.org/pypi/serving-utils
[release-image]: https://img.shields.io/github/release/Yoctol/serving-utils.svg


Some python utilities when using tensorflow-serving.

## Installation

From PYPI:
1. Manually install tensorflow CPU or GPU version.
2. `pip install serving-utils`

From Github repository:
1. `git clone git@github.com:Yoctol/serving-utils.git`
2. `make install`


## Usage

1. Saver
```python
import tensorflow as tf

from serving_utils import Saver

saver = Saver(
    session=tf.Session(graph=your_graph),
    output_dir='/path/to/serving',
    signature_def_map={
        'predict': tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={'input': tf.Tensor...},
            outputs={'output': tf.Tensor...},
        )
    },
)
saver.save(...)
```

2. Client
```python
from serving_utils import Client, PredictInput

client = Client(addr="localhost:8500")
client.predict(
    [PredictInput(name='input', value=np.ones(1, 10))],
    output_names=['output'],
    model_signature_name='predict',
)

# or async
await client.async_predict(...)
```

## Test

Run the following commands:
```
make lint
make test
```

## Dev

```
make install
```
