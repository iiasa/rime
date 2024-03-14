import os
import subprocess as sp
import time

def cdo(cmd, env=None):
    print(f"cdo {cmd}")
    my_env = dict(os.environ, **env) if env is not None else None
    return sp.check_call(f"cdo {cmd}", shell=True, env=my_env)


class Timer:
    def __init__(self):
        self.reset()
        self.total = 0

    def reset(self):
        self.t = time.time()

    def _check(self):
        t = time.time()
        delta = t - self.t
        self.total += delta
        return delta

    def check(self, message):
        delta = self._check()
        print(message, delta*1000, "ms")
        self.reset()

    def stop(self, message):
        self._check()
        print(message, self.total*1000, "ms")
        self.total = 0