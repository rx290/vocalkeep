from .rocm_env import apply_rocm_override

apply_rocm_override()  # must run before anything imports torch
