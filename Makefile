.DEFAULT_GOAL := all

.PHONY: install
install:
	pipenv install

.PHONY: install-dev
install-dev:
	pipenv install --dev

.PHONY: lint
lint:
	flake8

.PHONY: test
test:
	pytest -m "not integration" .

.PHONY: testall
testall:
	python train_for_test.py
	docker-compose up -d
	pytest .
	docker-compose down

.PHONY: all
all: install-dev lint testall

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
