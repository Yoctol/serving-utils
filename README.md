# serving-utils

[![Build Status](https://travis-ci.org/Yoctol/serving-utils.svg?branch=master)](https://travis-ci.org/Yoctol/serving-utils) [![PyPI version](https://badge.fury.io/py/serving-utils.svg)](https://badge.fury.io/py/serving-utils)

Some python utilities when using tensorflow-serving.

## Installation

1. Manually install tensorflow CPU or GPU version.
2. `pip install serving-client`

## Usage

```python
from serving_utils import Client

client = Client(addr="localhost:8500")
client.predict(...)

# or async
await client.async_predict(...)
```

## Test

Run the following commands:
```
make lint
make test
```
