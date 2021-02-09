format:
	black .
	isort .
	docformatter -i -r . --wrap-summaries 88 --wrap-descriptions 88

lint:
	env PYTHONPATH=. pytest --pylint --flake8 --mypy --ignore tests

test:
	env PYTHONPATH=. pytest tests --cov=kindle --cov-report term-missing --cov-report html

install:
	conda env update -p ${CONDA_PREFIX} --file environment.yml

dev:
	conda env update -p ${CONDA_PREFIX} --file environment.yml
	conda env update -p ${CONDA_PREFIX} --file environment-dev.yml
	pre-commit install --hook-type pre-commit --hook-type pre-push

doc:
	rm -rf docs/
	env PYTHONPATH=. pdoc -o docs --html --force kindle/
	mv docs/kindle/* docs/
	rmdir docs/kindle

push-doc:
	rm -rf docs/
	env PYTHONPATH=. pdoc -o docs --html --force kindle/
	mv docs/kindle/* docs/
	rmdir docs/kindle
	git add docs
	git add -u docs
	git commit --amend -C HEAD

build-pypi:
	rm dist/*
	python3 setup.py sdist bdist_wheel
	python3 -m twine check dist/*

dist-upload:
	python3 -m twine upload dist/*
