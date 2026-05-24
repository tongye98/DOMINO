import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .utils import BASE_IMPORTS, check_correctness


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def evaluate_score(args):
    gs, (c, i, o) = args
    execution_results = []
    for g in gs:
        if i in g:
            pass
        else:
            code_to_execute = f"{BASE_IMPORTS}\n{c}\nassert {o} == {g}"
            execution_results.append(check_correctness(code_to_execute, 3))
    if len(execution_results) == 0:
        execution_results = [False] * len(gs)
    return execution_results


def code_execution_metrics(samples, generations):
    references = [(doc["code"], doc["input"], doc["output"]) for doc in samples]
    with ProcessPoolExecutor() as executor:
        args_list = zip(generations, references)
        results = executor.map(evaluate_score, args_list)
    all_results = list(results)

    pass_at_1s = []
    for execution_result in all_results:
        c, n = execution_result.count(True), len(execution_result)
        pass_at_1s.append(pass_at_k(n, c, 1))
    metrics = {"pass@1": sum(pass_at_1s) / len(pass_at_1s) * 100}

    results_dict = {}
    for i, r in enumerate(all_results):
        r_new = []
        for _r in r:
            r_new.append([_r])
        results_dict[i] = r_new
    return [metrics, results_dict]
