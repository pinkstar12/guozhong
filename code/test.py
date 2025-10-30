"""Compatibility entry point forwarding to :mod:`airbattle.test`."""
from runpy import run_module
from airbattle.test import *  # noqa: F401,F403

if __name__ == "__main__":
    run_module("airbattle.test", run_name="__main__")
