"""Compatibility entry point forwarding to :mod:`airbattle.demo`."""
from runpy import run_module
from airbattle.demo import *  # noqa: F401,F403

if __name__ == "__main__":
    run_module("airbattle.demo", run_name="__main__")
