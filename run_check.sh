#!/bin/bash
#
# Shell script for formating, linting and unit test
#
# - Author: Jongkuk Lim
# - Contact: lim.jeikei@gmail.com

declare -A CMD_DESC

CMD_DESC[format]="Run formating"
CMD_DESC[lint]="Run linting check"
CMD_DESC[test]="Run unit test"
CMD_DESC[doc]="Generate MKDocs document"
CMD_DESC[doc-server]="Run MKDocs hosting server (in local)"
CMD_DESC[init-conda]="Create conda environment with default name"
CMD_DESC[init-precommit]="Install pre-commit plugin"
CMD_DESC[init]="Run init-conda and init-precommit"
CMD_DESC[all]="Run formating, linting and unit test"
CMD_DESC[clean]="Clean mypy and pytest cache."
CMD_DESC[build]="Build PyPi Package for kindle."
CMD_DESC[deploy]="Upload PyPi Package for kindle."

declare -A CMD_LIST

CMD_LIST[format]="black . && \
                  isort . && \
                  docformatter -i -r . --wrap-summaries 88 --wrap-descriptions 88"
CMD_LIST[lint]="env PYTHONPATH=. pytest --pylint --mypy --flake8 --ignore tests --ignore cpp"
CMD_LIST[test]="env PYTHONPATH=. pytest tests --cov=kindle --cov-report term-missing --cov-report html"
CMD_LIST[doc]="env PYTHONPATH=. mkdocs build --no-directory-urls"
CMD_LIST[doc-server]="env PYTHONPATH=. mkdocs serve -a 127.0.0.1:8000 --no-livereload"
CMD_LIST[init-conda]="conda env create -f environment.yml"
CMD_LIST[init-dev]="conda env update -f environment-dev.yml"
CMD_LIST[init-precommit]="pre-commit install --hook-type pre-commit --hook-type pre-push"
CMD_LIST[init]="${CMD_LIST[init-conda]} && ${CMD_LIST[init-dev]} && ${CMD_LIST[init-precommit]}"
CMD_LIST[all]="${CMD_LIST[format]} && ${CMD_LIST[lint]} && ${CMD_LIST[test]}"
CMD_LIST[clean]="rm -r .mypy_cache .pytest_cache"
CMD_LIST[build]="mkdir -p dist && \
                 touch dist/empty && \
                 rm dist/* && \
                 rm dist/empty && \
                 python3 setup.py sdist bdist_wheel && \
                 python3 -m twine check dist/*"
CMD_LIST[deploy]="python3 -m twine upload dist/*"

exitCode=0
for _arg in $@
do
    if [[ ${CMD_LIST[$_arg]} == "" ]]; then
        echo "$_arg is not valid option!"
        echo "--------------- $0 Usage ---------------"
        for _key in "${!CMD_LIST[@]}"
        do
            echo "$0 $_key - ${CMD_DESC[$_key]}"
        done
        exit 0
    else
        echo "Run ${CMD_LIST[$_arg]}"
        eval ${CMD_LIST[$_arg]}

        result=$?
        if [ $result -ne 0 ]; then
            exitCode=$result
        fi
    fi
done

exit $exitCode
