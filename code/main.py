"""Compatibility entry point forwarding to :mod:`airbattle.main`."""
from runpy import run_module
from airbattle.main import *  # noqa: F401,F403

if __name__ == "__main__":
    run_module("airbattle.main", run_name="__main__")
