"""Legacy compatibility wrapper for the former `code` package."""
from importlib import import_module
import sys

_new_pkg = import_module("airbattle")

# Re-export public attributes from the unified package
globals().update({k: getattr(_new_pkg, k) for k in getattr(_new_pkg, "__all__", [])})

_REDIRECTS = {
    "code.AircraftManeuvering": "airbattle.AircraftManeuvering",
    "code.DroneCombatSystem": "airbattle.DroneCombatSystem",
    "code.Hierarchical": "airbattle.Hierarchical",
    "code.BattlefieldEnvironment": "airbattle.BattlefieldEnvironment",
    "code.BattlefieldEnvironment.BattlefieldEnvironment": "airbattle.BattlefieldEnvironment.BattlefieldEnvironment",
    "code.StrategyPredictor": "airbattle.StrategyPredictor",
    "code.StrategyPredictor.StrategyPredictor": "airbattle.StrategyPredictor.StrategyPredictor",
    "code.ThreatAnalyzer": "airbattle.ThreatAnalyzer",
    "code.ThreatAnalyzer.ThreatAnalyzer": "airbattle.ThreatAnalyzer.ThreatAnalyzer",
    "code.demo": "airbattle.demo",
    "code.main": "airbattle.main",
    "code.test": "airbattle.test",
}

for legacy_name, unified_name in _REDIRECTS.items():
    if legacy_name not in sys.modules:
        sys.modules[legacy_name] = import_module(unified_name)

__all__ = list(getattr(_new_pkg, "__all__", []))
