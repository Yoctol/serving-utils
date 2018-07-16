.DEFAULT_GOAL := all

.PHONY: install
install:
	pip install -U pip wheel setuptools
	pip install -r requirements.txt
	pip install -e .

.PHONY: lint
lint:
	flake8

.PHONY: test
test:
	pytest .

.PHONY: testall
testall:
	python train_for_test.py
	docker-compose up -d
	pytest .
	docker-compose stop
	docker-compose rm -f

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
	rm -r test-fixtures/
	make -C docs clean
	python setup.py clean
