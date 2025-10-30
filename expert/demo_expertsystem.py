"""Compatibility stub forwarding to :mod:`airbattle.expert.demo_expertsystem`."""
from runpy import run_module
from airbattle.expert.demo_expertsystem import *  # noqa: F401,F403

if __name__ == "__main__":
    run_module("airbattle.expert.demo_expertsystem", run_name="__main__")
