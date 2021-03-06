# Contributing to SPECTR

### Please have the following VSCode extensions setup:
- Python Docstring Generator
    - Type `"""` and `Enter` to generate a template docstring for your function
- Flake8 Linter
    - `Cntrl + Shift + P`, type `Select Linter`, select `flake8`

Have the following setup in `.vscode/settings.json`:
```
{
    "python.linting.flake8Enabled": true,
    "python.linting.enabled": true,
    "editor.rulers": [
            120
    ],
    "python.linting.flake8Args": [
        "--max-line-length=120",
        "--ignore = E501",
    ],
}
```


### To contribute to new code:
- Clone repository to your machine
- Create a branch for your changes
    - `$ git checkout -b <name of your branch>`
- Reformat your files using Python black
    - `$black <file name>.py -l 120`
- Ensure static analysis with flake8 does not throw any errors
- Add docstrings for every function
- Open a PR to merge into `main` branch
