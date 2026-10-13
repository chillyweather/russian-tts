from rutts.engines.base import get_engine, register_engine
from rutts.engines.fish_mlx import FishMlxEngine
from rutts.engines.qwen_mlx import QwenMlxEngine
from rutts.engines.voxcpm2 import VoxCpm2Engine

register_engine("qwen-mlx", QwenMlxEngine)
register_engine("fish-mlx", FishMlxEngine)
register_engine("voxcpm2", VoxCpm2Engine)

__all__ = ["FishMlxEngine", "QwenMlxEngine", "VoxCpm2Engine", "get_engine", "register_engine"]
