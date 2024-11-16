init-e:
	python -m pip install -e .

init:
	python -m pip install .

check:
	ruff check 

format:
	ruff format

lint:
	python -m pylint pytens
	python -m flake8 pytens

type-check:
	python -m mypy 

test:
	python -m unittest tests/main_test.py -v

ci: check format lint type-check test

.PHONY: init check format lint test type-check
