# 1. Installation
1. Clone repository
```shell
$ git clone https://github.com/JeiKeiLim/kindle.git
$ cd kindle
```

2. Install conda environment and dependencies

- Below command will create new conda environment (torch-kindle), install dependencies, and setup pre-commit hooks.

```shell
./run_check.sh init
```

# 3. Code style guide
- We follow conventional [PEP8](https://www.python.org/dev/peps/pep-0008/) guideline.

# 4. Code check tool
## 4.1. Formating
```shell
./run_check.sh format
```

## 4.2. Linting
```shell
./run_check.sh lint
```

## 4.3. Unit test
```shell
./run_check.sh test
```

## 4.4. You can add check procedures as many as you want.
```shell
./run_check.sh format lint
./run_check.sh format lint test
```

# 5. Commit
* **DO NOT** commit on `main` branch. Make a new branch and commit and create a PR request.
* Formatting and linting is auto-called when you `commit` and `push` but we advise you to run `./run_check all` occasionally.

# 5. Documentation
## 4.1. Generate document
```shell
./run_check doc
```

## 4.2. Run local documentation web server
```shell
./run_check doc-server
```

