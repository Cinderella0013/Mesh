"""Disjoint measured phase costs; retain raw zero CPU samples."""
import time


def timed(call):
    elapsed_start=time.perf_counter_ns();cpu_start=time.process_time_ns()
    result=call()
    return result,time.perf_counter_ns()-elapsed_start,time.process_time_ns()-cpu_start


def measure(call):
    result,elapsed,cpu=timed(call)
    return result,dict(elapsed_ns=elapsed,cpu_ns=cpu)
