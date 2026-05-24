import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile


BASE_IMPORTS = """from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
from string import ascii_lowercase
from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge
from functools import reduce, cache, lru_cache
from random import randrange, shuffle
from operator import itemgetter, sub
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator
import copy
import string
import math
import collections
import bisect
import heapq
import functools
import random
import itertools
import operator
import re
import numpy as np
import pandas as pd
from math import log, prod
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle
from functools import lru_cache, reduce, partial
from operator import iand
import sys
"""


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    yield
    os.chdir(cwd)


def reliability_guard(maximum_memory_bytes=None):
    """
    Disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem contents, etc.)
    """
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None

    sys.displayhook = None

    import signal

    signal.signal = None


def check_correctness(code_to_execute, timeout=3):
    """
    Check correctness of code execution with a timeout.
    Adapted from OpenAI's HumanEval.
    """
    def unsafe_execute():
        with contextlib.redirect_stdout(None):
            try:
                exec_globals = {}
                exec(code_to_execute, exec_globals)
                result["passed"] = True
                result["result"] = "execution successful"
            except Exception as e:
                result["passed"] = False
                result["result"] = str(e)

    manager = multiprocessing.Manager()
    result = manager.dict()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.kill()
        return {"passed": False, "result": "timeout"}

    return dict(result)
