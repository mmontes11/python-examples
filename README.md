# python-examples

Miscellaneous Python examples for learning purposes. All of them share a single [uv](https://docs.astral.sh/uv/) project and virtualenv.

## Usage

```bash
uv sync                                # create/update .venv
uv run examples/pytorch.py             # run an example
uv add <package>                       # add a dependency an example needs
```