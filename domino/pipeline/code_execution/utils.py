# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from math import log, prod
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle
from functools import lru_cache, reduce, partial
from operator import iand
import sys
"""


def check_correctness(code_to_execute, timeout=3):
    """
    Check correctness of code execution with a timeout.
    Returns dict with 'passed' (bool) and 'result' (str).
    """
    manager = multiprocessing.Manager()
    result = manager.dict()
    
    def execute_code(code, result):
        try:
            exec(code, globals())
            result['passed'] = True
            result['result'] = 'execution successful'
        except Exception as e:
            result['passed'] = False
            result['result'] = str(e)
    
    p = multiprocessing.Process(target=execute_code, args=(code_to_execute, result))
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return {'passed': False, 'result': 'timeout'}
    
    return dict(result)


def postsynthetic_extract(generated_text):
    """Extract code and answer from generated text."""
    import re
    pattern = r"""
        \[PYTHON\]   
        (.*?)         
        \[/PYTHON]   
        .*?           
        \[ANSWER\]    
        (.*?)         
        \[/ANSWER]   
    """
    match = re.search(pattern, generated_text, re.DOTALL | re.VERBOSE)

    code = match.group(1).strip() if match else None
    execution_answer = match.group(2).strip() if match else None

    if code is None or execution_answer is None:
        return (None, None)
    
    code = f"[PYTHON]\n{code}\n[/PYTHON]"
    execution_answer = f"[ANSWER]\n{execution_answer}\n[/ANSWER]"

    return (code, execution_answer)
