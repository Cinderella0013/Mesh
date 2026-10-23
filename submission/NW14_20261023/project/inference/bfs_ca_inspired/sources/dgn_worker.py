"""Isolated Python3.7 CPU worker for the preserved raw DGN/common11 graph."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time

HERE=Path(__file__).resolve().parent


def build(sources,seed,stage_path=None):
    if sys.version_info[:2]!=(3,7) or struct.calcsize('P')!=8 or not __debug__ or not sys.flags.isolated:
        raise RuntimeError('isolated Python3.7 x64 required')
    if sys.prefix==sys.base_prefix or 'include-system-site-packages = false' not in (Path(sys.prefix)/'pyvenv.cfg').read_text().lower():
        raise RuntimeError('isolated native venv required')
    stage_path=HERE.parent/'dgn_routing_execution_20261003/model_stage.py' if stage_path is None else Path(stage_path).resolve()
    spec=importlib.util.spec_from_file_location('existing_dgn_stage',str(stage_path))
    stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
    versions={p['name'].lower():p['version'] for p in stage.helpers.probe.installed_packages(sys.path)}
    for name,version in dict(stage.helpers.probe.PINS,h5py='2.10.0',protobuf='3.20.3').items():
        if versions.get(name)!=version:raise RuntimeError('native package pin mismatch: '+name)
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['KERAS_BACKEND']='tensorflow'
    os.environ['OMP_NUM_THREADS']='1';os.environ['TF_NUM_INTRAOP_THREADS']='1';os.environ['TF_NUM_INTEROP_THREADS']='1'
    import numpy as np
    import tensorflow as tf
    if tf.config.get_visible_devices('GPU'):raise RuntimeError('native GPU visible')
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    np.random.seed(seed);tf.random.set_seed(seed)
    sources=Path(sources).resolve();manifest=json.loads((sources/'source_manifest.json').read_text())
    if manifest['schema']!='dgn-common-source-v1' or manifest['feature_width']!=11 or manifest['agents']!=15:
        raise ValueError('common11 DGN source identity required')
    for name,expected in manifest['files'].items():
        if Path(name).name!=name or hashlib.sha256((sources/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('DGN source changed: '+name)
    ns=dict(np=np,tf=tf,n_data=15,neighbors=4,len_feature=11,action_space=4)
    with tf.device('/CPU:0'):
        for name in ('model_definitions.py','online_graph.py','target_graph.py','tau.py','gamma.py','regularizer_graph.py'):
            stage.execute(sources,name,ns)
    model,target,regularizer=ns['model'],ns['model_t'],ns['model_r']
    if len(model.weights)!=22 or len(target.weights)!=22 or len(regularizer.weights)!=16:
        raise ValueError('native graph sharing/parameter count changed')
    if any(model.inputs[2*j].shape.as_list()!=[None,11] for j in range(15)):
        raise ValueError('native observation width mismatch')
    if set(id(w) for w in model.weights)&set(id(w) for w in target.weights):raise ValueError('online/target alias')
    if not set(id(w) for w in regularizer.weights)<=set(id(w) for w in model.weights):raise ValueError('regularizer sharing changed')
    return ns,stage


def weights_hash(np,values):
    digest=hashlib.sha256()
    for value in values:
        if not np.isfinite(value).all():raise ValueError('nonfinite DGN parameters')
        digest.update(str(value.shape).encode());digest.update(str(value.dtype).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def load_weights(np,model,path,expected=None,checkpoint_sha256=None):
    path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if checkpoint_sha256 is not None and digest!=checkpoint_sha256:
        raise ValueError('DGN inference checkpoint bytes changed')
    with np.load(str(path),allow_pickle=False) as saved:
        if (saved['schema'].item()!='dgn-common11-inference-v1' or
            set(saved.files)!={'schema',*('weight_'+str(i) for i in range(22))}):
            raise ValueError('wrong DGN checkpoint schema/coverage')
        values=[saved['weight_'+str(i)] for i in range(22)]
    if any(a.shape!=b.shape or a.dtype!=b.dtype for a,b in zip(values,model.get_weights())):
        raise ValueError('DGN checkpoint shape/dtype mismatch')
    identity=weights_hash(np,values)
    if expected is not None and expected!=identity:raise ValueError('DGN replacement model identity differs')
    model.set_weights(values)
    if weights_hash(np,model.get_weights())!=identity:raise ValueError('DGN loaded parameters differ')
    return identity,digest


def runtime_identity(stage):
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    process=ctypes.c_size_t();system=ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),ctypes.byref(process),ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    tf=sys.modules['tensorflow']
    runtime=dict(python=sys.version,executable=sys.executable,base_executable=sys._base_executable,
        executable_sha256=sha(sys.executable),base_executable_sha256=sha(sys._base_executable),
        venv_config_sha256=sha(Path(sys.prefix)/'pyvenv.cfg'),device='cpu',isolated=bool(sys.flags.isolated),pid=os.getpid(),
        process_affinity=dict(process_mask=process.value,system_mask=system.value),
        intra_threads=tf.config.threading.get_intra_op_parallelism_threads(),
        inter_threads=tf.config.threading.get_inter_op_parallelism_threads(),
        environment={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS')},
        package_versions={p['name'].lower():p['version'] for p in stage.helpers.probe.installed_packages(sys.path)})
    files={str(Path(m.__file__).resolve()):sha(m.__file__) for m in list(sys.modules.values())
        if getattr(m,'__file__',None) and Path(m.__file__).is_file()}
    return runtime,files


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sources',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--weights');parser.add_argument('--stage-path');args=parser.parse_args()
    setup=time.process_time_ns();ns,stage=build(args.sources,args.seed,stage_path=args.stage_path)
    np,model=ns['np'],ns['model']
    if args.weights:
        load_weights(np,model,args.weights)
    identity=weights_hash(np,model.get_weights())
    warm_obs=np.zeros((1,15,11),dtype='float32');warm_masks=np.zeros((1,15,4,15),dtype='float32')
    for i in range(15):
        for slot in range(4):warm_masks[0,i,slot,(i+slot)%15]=1
    model.predict(stage.model_inputs(np,warm_obs,warm_masks))
    runtime,dependencies=runtime_identity(stage)
    ready=dict(feature_width=11,online_weights=22,target_weights=22,regularizer_weights=16,
               aggregation='UPSTREAM_RAW_SCORES',optimizer_steps=0,model_sha256=identity,
               setup_and_zero_fixture_warmup_cpu_ns=time.process_time_ns()-setup,
               runtime=runtime,runtime_dependencies=dependencies,seed=args.seed,gamma=float(ns['GAMMA']),tau=float(ns['TAU']))
    print('DGN_READY '+json.dumps(ready,allow_nan=False),flush=True)
    for line in sys.stdin:
        started=time.process_time_ns();request=json.loads(line)
        op=request.get('op')
        if op=='exit':return
        if op=='save':
            path=Path(request['path']).resolve()
            if HERE not in path.parents or path.exists():raise ValueError('fresh checkpoint inside comparison root required')
            np.savez(path,schema=np.array('dgn-common11-inference-v1'),**{'weight_'+str(i):w for i,w in enumerate(model.get_weights())})
            response=dict(op=op,model_sha256=identity,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        elif op=='refresh':
            if request['previous_model_sha256']!=identity:raise ValueError('stale DGN replacement request')
            previous=identity
            identity,digest=load_weights(np,model,request['path'],request['model_sha256'],request['checkpoint_sha256'])
            response=dict(op=op,previous_model_sha256=previous,model_sha256=identity,
                checkpoint_sha256=digest,cpu_ns=time.process_time_ns()-started,optimizer_steps=0)
        elif op=='predict':
            state=np.asarray(request['state'],dtype='float32');masks=np.asarray(request['masks'],dtype='float32')
            if (state.shape!=(15,11) or not np.isfinite(state).all() or masks.shape!=(15,4,15) or
                not np.isfinite(masks).all() or not np.all((masks==0)|(masks==1)) or not np.all(masks.sum(-1)==1)):
                raise ValueError('invalid complete common11 state/neighborhood mask')
            inputs=stage.model_inputs(np,state[None],masks[None])
            neural_elapsed=time.perf_counter_ns();neural_cpu=time.process_time_ns()
            q=np.asarray(model.predict(inputs))[:,0,:]
            neural=dict(elapsed_ns=time.perf_counter_ns()-neural_elapsed,cpu_ns=time.process_time_ns()-neural_cpu)
            guard_elapsed=time.perf_counter_ns();guard_cpu=time.process_time_ns()
            if q.shape!=(15,4) or not np.isfinite(q).all():raise ValueError('nonfinite native DGN Q')
            if weights_hash(np,model.get_weights())!=identity:raise ValueError('inference changed DGN parameters')
            validation=dict(elapsed_ns=time.perf_counter_ns()-guard_elapsed,cpu_ns=time.process_time_ns()-guard_cpu)
            response=dict(op=op,q=q.tolist(),model_sha256=identity,cpu_ns=time.process_time_ns()-started,
                phase_costs=dict(neural=neural,validation=validation))
        else:raise ValueError('unknown DGN worker operation')
        print('DGN_RESPONSE '+json.dumps(response,allow_nan=False),flush=True)


if __name__=='__main__':main()
