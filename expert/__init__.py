"""Legacy compatibility wrapper for the former `expert` package."""
from importlib import import_module
import sys

_new_pkg = import_module("airbattle.expert")

globals().update({k: getattr(_new_pkg, k) for k in getattr(_new_pkg, "__all__", [])})

_REDIRECTS = {
    "expert.attack_expertsystem": "airbattle.expert.attack_expertsystem",
    "expert.defend_expertsystem": "airbattle.expert.defend_expertsystem",
    "expert.demo_expertsystem": "airbattle.expert.demo_expertsystem",
    "expert.expertsystem": "airbattle.expert.expertsystem",
    "expert.generators": "airbattle.expert.generators",
    "expert.utils": "airbattle.expert.utils",
    "expert.behavior_tree": "airbattle.expert.behavior_tree",
    "expert.behavior_tree.engine": "airbattle.expert.behavior_tree.engine",
    "expert.behavior_tree.nodes": "airbattle.expert.behavior_tree.nodes",
    "expert.optimize": "airbattle.expert.optimize",
    "expert.optimize.DE": "airbattle.expert.optimize.DE",
    "expert.optimize.NSGA": "airbattle.expert.optimize.NSGA",
    "expert.optimize.explanation_generator": "airbattle.expert.optimize.explanation_generator",
    "expert.evalthreat": "airbattle.expert.evalthreat",
    "expert.evalthreat.SingleAir_to_MulitMissile": "airbattle.expert.evalthreat.SingleAir_to_MulitMissile",
    "expert.evalthreat.Mutilair_to_Mutilair": "airbattle.expert.evalthreat.Mutilair_to_Mutilair",
    "expert.evalthreat.AirToAirThreatModels": "airbattle.expert.evalthreat.AirToAirThreatModels",
    "expert.evalthreat.AirToAirThreatModels.BaseAttackArea": "airbattle.expert.evalthreat.AirToAirThreatModels.BaseAttackArea",
    "expert.evalthreat.AirToAirThreatModels.BeyondVisualRangeThreatModel": "airbattle.expert.evalthreat.AirToAirThreatModels.BeyondVisualRangeThreatModel",
    "expert.evalthreat.AirToAirThreatModels.test_base_attack_area": "airbattle.expert.evalthreat.AirToAirThreatModels.test_base_attack_area",
    "expert.evalthreat.AirToAirThreatModels.test_beyond_visual_range_threat_model": "airbattle.expert.evalthreat.AirToAirThreatModels.test_beyond_visual_range_threat_model",
    "expert.evalthreat.AirToMissileThreatModels": "airbattle.expert.evalthreat.AirToMissileThreatModels",
    "expert.evalthreat.AirToMissileThreatModels.ClassicMethod": "airbattle.expert.evalthreat.AirToMissileThreatModels.ClassicMethod",
}

for legacy_name, unified_name in _REDIRECTS.items():
    if legacy_name not in sys.modules:
        sys.modules[legacy_name] = import_module(unified_name)

__all__ = list(getattr(_new_pkg, "__all__", []))
