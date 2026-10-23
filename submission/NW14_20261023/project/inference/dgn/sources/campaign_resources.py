"""Prospective per-process resources; no system/user-wide affinity changes."""
import ctypes
import os
import platform
import sys
from native_fixture_runner import LOADER,SYSTEM_LIBS


def affinity():
    if sys.platform!='win32':raise ValueError('qualified Windows controller host required')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    process=ctypes.c_size_t();system=ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),ctypes.byref(process),ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(process_mask=process.value,system_mask=system.value)


def apply():
    initial=affinity()
    if not initial['process_mask']&4:raise ValueError('declared controller logical CPU2 unavailable')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),4):raise ctypes.WinError(ctypes.get_last_error())
    if affinity()['process_mask']!=4:raise ValueError('actual controller affinity differs from protocol')
    environment=dict(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
    os.environ.update(environment)
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:raise ValueError('actual Torch thread count differs')
    return dict(affinity=affinity(),initial_affinity=initial,environment=environment,
        torch_intraop_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
        torch_version=torch.__version__,python=sys.version,platform=platform.platform(),
        cpu_count=os.cpu_count(),device='cpu',wsl_vcpu=2,
        mapping_limitation='WSL vCPU2 does not prove host physical-core identity')


def wsl_prefix():
    # The recovered ELF loader is required for both taskset and Python/native
    # executables. taskset changes only its own process and descendants.
    return ['wsl','-d','Ubuntu','--exec',LOADER,'--library-path',SYSTEM_LIBS,
        '/usr/bin/taskset','-c','2',LOADER,'--library-path',SYSTEM_LIBS]


def check_profile(profile):
    import torch
    if (not isinstance(profile,dict) or profile.get('wsl_vcpu')!=2 or affinity()['process_mask']!=4
        or profile.get('affinity',{}).get('process_mask')!=4 or torch.get_num_threads()!=1
        or torch.get_num_interop_threads()!=1 or not profile.get('environment')
        or any(os.environ.get(key)!=value for key,value in profile['environment'].items())):
        raise ValueError('actual controller resources differ from declared profile')
    return profile


def native_command(command):
    # taskset execs the already declared native loader command, preserving its
    # library-path/inhibit-cache settings without a second library-path option.
    return wsl_prefix()[:-3]+list(command)
