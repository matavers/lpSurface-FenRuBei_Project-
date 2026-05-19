"""
Core 包：延迟导入，避免在 ``import core.meshProcessor`` 等子模块时
加载 CGAL / 重型依赖导致访问冲突。
"""

from typing import Any

__all__ = [
    'MeshProcessor',
    'NonSphericalTool',
    'AdvancedSurfacePartitioner',
    'ToolOrientationField',
    'IsoScallopFieldGenerator',
    'PathGenerator',
    'IndicatorCalculator',
    'NURBSProcessor',
]

_LAZY_MODULES = {
    'MeshProcessor': ('.meshProcessor', 'MeshProcessor'),
    'NonSphericalTool': ('.nonSphericalTool', 'NonSphericalTool'),
    'AdvancedSurfacePartitioner': ('.advancedSurfacePartitioner', 'AdvancedSurfacePartitioner'),
    'ToolOrientationField': ('.toolOrientationField', 'ToolOrientationField'),
    'IsoScallopFieldGenerator': ('.isoScallopField', 'IsoScallopFieldGenerator'),
    'PathGenerator': ('.pathGenerator', 'PathGenerator'),
    'IndicatorCalculator': ('.indicatorCalculator', 'IndicatorCalculator'),
    'NURBSProcessor': ('.nurbsProcessor', 'NURBSProcessor'),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        import importlib
        mod_path, attr = _LAZY_MODULES[name]
        mod = importlib.import_module(mod_path, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(list(globals().keys()) + __all__)
