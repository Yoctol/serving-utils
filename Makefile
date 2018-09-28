.DEFAULT_GOAL := all

.PHONY: install
install:
	pip install pipenv==v2018.05.18
	pipenv run pip install -U Cython
	pipenv install --dev

.PHONY: lint
lint:
	pipenv run python -m flake8

.PHONY: test
test:
	pipenv run pytest -m "not integration" .

.PHONY: testall
testall:
	pipenv run python train_for_test.py
	docker-compose up -d
	pytest .
	docker-compose stop
	docker-compose rm -f
	rm -r .fake-models/

.PHONY: all
all: install testall lint

.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -rf .cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -r .fake-models/
	python setup.py clean
