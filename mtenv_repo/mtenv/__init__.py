# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
__version__ = "1.0"

__all__ = ["MTEnv", "make"]


def __getattr__(name):
    # Avoid importing gym-dependent modules at package import time.
    if name == "MTEnv":
        from mtenv.core import MTEnv

        return MTEnv
    if name == "make":
        from mtenv.envs.registration import make

        return make
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
