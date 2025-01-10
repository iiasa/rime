import os
import subprocess as sp
import time

def check_call(cmd, env=None, dry_run=False):
    print(cmd)
    if dry_run:
        return 0
    my_env = dict(os.environ, **env) if env is not None else None
    return sp.check_call(cmd, shell=True, env=my_env)

def cdo(cmd, **kwargs):
    return check_call(f"cdo {cmd}", **kwargs)

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
