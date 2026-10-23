"""Per-run inference weights/config plus exact source snapshots, without pickled code."""
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
import shutil
import uuid
from types import SimpleNamespace
import numpy as np
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(folder):return {p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*')
    if p.is_file() and p!=folder/'artifact_hashes.json'}


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def freeze(policy,output):
    import torch
    from policy_adapters import model_hash
    output=Path(output).resolve()
    if HERE not in output.parents or output.exists():raise ValueError('fresh owned policy capsule required')
    kind=type(policy).__name__
    if kind=='BehaviorPolicy':
        metadata=freeze(policy.base,output)
        copied=output/'sources/behavior_policy.py';copied.write_bytes((HERE/'behavior_policy.py').read_bytes())
        metadata['source_sha256']['behavior_policy.py']=sha(copied)
        metadata['behavior']=policy.checkpoint()
        write(output/'metadata.json',metadata);write(output/'artifact_hashes.json',inventory(output))
        return metadata
    if kind=='DGN':
        metadata=check(policy.capsule_path)
        if metadata['model_sha256']!=policy.identity:raise ValueError('DGN capsule/native identity differs')
        shutil.copytree(policy.capsule_path,output);return metadata
    if kind not in ('GATRAES','GANet','Diagnostic'):raise ValueError('qualified frozen policy required')
    if kind!='Diagnostic' and model_hash(policy.model)!=policy.identity:raise ValueError('policy weights changed before run')
    output.mkdir(parents=True)
    names=['policy_capsule.py','policy_adapters.py','policy_timing.py','ganet.py','candidate_sources/marl_gnn.py','candidate_sources/raes.py','candidate_sources/controls.py',
        'live_runtime.py','native_fixture_runner.py','common_observation.py','native_report.py']
    for name in names:
        copied=output/'sources'/name;copied.parent.mkdir(parents=True,exist_ok=True);copied.write_bytes((HERE/name).read_bytes())
    if kind=='Diagnostic':
        metadata=dict(schema='threeway-inference-capsule-v1',kind=kind,model_sha256=policy.identity,
            config=policy.config,random_state=policy.rng.bit_generator.state,
            source_sha256={name:sha(output/'sources'/name) for name in names},
            runtime=dict(python=sys.version,numpy=np.__version__,device='cpu'))
        write(output/'metadata.json',metadata);write(output/'artifact_hashes.json',inventory(output));return metadata
    if kind=='GATRAES':
        config=dict(input_dim=11,hidden_dim=64,output_dim=4,num_heads=4,temperature=float(policy.model.gat.temperature),
            dropout=policy.model.gat.attn_dropout.p,edge_dim=0)
    else:config=dict(agents=policy.model.agents,observation_dim=policy.model.observation_dim,actions=policy.model.actions)
    state=policy.model.state_dict()
    with (output/'weights.npz').open('xb') as stream:
        np.savez(stream,**{'state_'+str(i):value.detach().cpu().numpy().copy() for i,value in enumerate(state.values())})
    metadata=dict(schema='threeway-inference-capsule-v1',kind=kind,model_sha256=policy.identity,
        config=config,coefficients=getattr(policy,'coefficients',None),state_names=list(state),
        source_sha256={name:sha(output/'sources'/name) for name in names},
        runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,device='cpu'))
    if kind=='GATRAES':metadata['executor']=dict(executor=policy.executor,batches=policy.batches)
    write(output/'metadata.json',metadata);write(output/'artifact_hashes.json',inventory(output))
    return metadata


@contextmanager
def sources(folder):
    names=['policy_adapters','policy_timing','ganet','candidate_sources','candidate_sources.marl_gnn','candidate_sources.raes']
    if (folder/'candidate_sources/controls.py').is_file():names+=['candidate_sources.controls']
    if (folder/'behavior_policy.py').is_file():names+=['behavior_policy']
    if (folder/'live_runtime.py').is_file():names+=['live_runtime','native_fixture_runner','common_observation','native_report']
    prior={name:sys.modules.pop(name,None) for name in names};path=list(sys.path);sys.path.insert(0,str(folder))
    try:yield {name:importlib.import_module(name) for name in names}
    finally:
        sys.path[:]=path
        for name in names:
            sys.modules.pop(name,None)
            if prior[name] is not None:sys.modules[name]=prior[name]


def check(folder):
    folder=Path(folder).resolve();manifest=json.loads((folder/'artifact_hashes.json').read_text())
    if inventory(folder)!=manifest:raise ValueError('inference capsule inventory changed')
    metadata=json.loads((folder/'metadata.json').read_text())
    if metadata['schema']!='threeway-inference-capsule-v1':raise ValueError('qualified inference capsule schema required')
    for name,digest in metadata['source_sha256'].items():
        path=folder/'sources'/name
        if (folder/'sources').resolve() not in path.resolve().parents or sha(path)!=digest:raise ValueError('policy source changed')
    return metadata


def restore(folder,*,output=None):
    import torch
    folder=Path(folder).resolve();metadata=check(folder)
    with sources(folder/'sources') as bundle,torch.random.fork_rng(devices=[]):
        adapters=bundle['policy_adapters']
        if metadata['kind']=='Diagnostic':
            policy=adapters.Diagnostic(**metadata['config'])
            if policy.identity!=metadata['model_sha256']:raise ValueError('diagnostic configuration identity differs')
            policy.rng.bit_generator.state=metadata['random_state']
            return policy
        if metadata['kind']=='DGN':
            if output is None:output=HERE/'evidence'/('restored_dgn_capsule_'+uuid.uuid4().hex[:12])
            policy=adapters.DGN(capsule=folder,output=output,owner_root=HERE)
            if 'behavior' in metadata:policy=bundle['behavior_policy'].BehaviorPolicy.restore(policy,metadata['behavior'])
            return policy
        if metadata['kind']=='GATRAES':model=bundle['candidate_sources.marl_gnn'].GAT_Q_Network(**metadata['config'])
        elif metadata['kind']=='GANet':model=bundle['ganet'].LocalActors(**metadata['config'])
        else:raise ValueError('unknown frozen inference policy')
        expected=model.state_dict()
        if list(expected)!=metadata['state_names']:raise ValueError('policy state names differ')
        with np.load(folder/'weights.npz',allow_pickle=False) as saved:
            if set(saved.files)!={'state_'+str(i) for i in range(len(expected))}:raise ValueError('policy state coverage differs')
            loaded={}
            for i,(name,value) in enumerate(expected.items()):
                array=saved['state_'+str(i)]
                if array.shape!=tuple(value.shape) or array.dtype!=value.numpy().dtype or not np.isfinite(array).all():
                    raise ValueError('policy state shape/dtype/finite values differ')
                loaded[name]=torch.from_numpy(array.copy())
        model.load_state_dict(loaded,strict=True)
        if metadata['kind']=='GATRAES':policy=adapters.GATRAES(model,**metadata['coefficients'],**metadata.get('executor',{}))
        else:policy=adapters.GANet(SimpleNamespace(config=metadata['config'],target_actors=model))
        if policy.identity!=metadata['model_sha256']:raise ValueError('policy weights/config identity differs')
        if 'behavior' in metadata:policy=bundle['behavior_policy'].BehaviorPolicy.restore(policy,metadata['behavior'])
    return policy


def snapshot_dgn(graphs,weights,seed,output):
    output=Path(output).resolve()
    if HERE not in output.parents or output.exists():raise ValueError('fresh owned native DGN capsule required')
    output.mkdir(parents=True)
    names=['policy_capsule.py','policy_adapters.py','policy_timing.py','ganet.py','dgn_worker.py',
        'candidate_sources/marl_gnn.py','candidate_sources/raes.py','candidate_sources/controls.py','live_runtime.py','native_fixture_runner.py',
        'common_observation.py','native_report.py']
    for name in names:
        copied=output/'sources'/name;copied.parent.mkdir(parents=True,exist_ok=True);copied.write_bytes((HERE/name).read_bytes())
    historical=['dgn_routing_execution_20261003/model_stage.py']+['dgn_cpu_preparation_20261001/'+name for name in
        ('routing_target_probe.py','routing_invariants_probe.py','routing_full_model_probe.py','routing_repair_probe.py','routing_td_probe.py','routing_probe.py')]
    for name in historical:
        copied=output/'sources/historical'/name;copied.parent.mkdir(parents=True,exist_ok=True);copied.write_bytes((HERE.parent/name).read_bytes())
    shutil.copytree(Path(graphs),output/'graphs');(output/'weights.npz').write_bytes(Path(weights).read_bytes())
    digest=hashlib.sha256()
    with np.load(output/'weights.npz',allow_pickle=False) as saved:
        if saved['schema'].item()!='dgn-common11-inference-v1' or set(saved.files)!={'schema',*('weight_'+str(i) for i in range(22))}:
            raise ValueError('qualified native inference export required')
        for i in range(22):
            value=saved['weight_'+str(i)]
            if not np.isfinite(value).all():raise ValueError('nonfinite native DGN weights')
            digest.update(str(value.shape).encode());digest.update(str(value.dtype).encode());digest.update(value.tobytes())
    metadata=dict(schema='threeway-inference-capsule-v1',kind='DGN',seed=seed,model_sha256=digest.hexdigest(),runtime=None,
        source_sha256={p.relative_to(output/'sources').as_posix():sha(p) for p in (output/'sources').rglob('*') if p.is_file()})
    write(output/'metadata.json',metadata);write(output/'artifact_hashes.json',inventory(output));return metadata


def bind_runtime(folder,ready):
    metadata=check(folder)
    if ready['model_sha256']!=metadata['model_sha256']:raise ValueError('native source/weights identity differs')
    metadata['runtime']=ready['runtime'];metadata['runtime_dependencies']=ready['runtime_dependencies']
    write(folder/'metadata.json',metadata);write(folder/'artifact_hashes.json',inventory(folder))


def refresh_dgn(folder,weights,output):
    """New weights with the actual running worker's source/runtime provenance."""
    folder=Path(folder).resolve();output=Path(output).resolve();weights=Path(weights).resolve()
    metadata=check(folder)
    if metadata['kind']!='DGN' or HERE not in output.parents or output.exists():
        raise ValueError('fresh owned replacement DGN capsule required')
    digest=hashlib.sha256()
    with np.load(weights,allow_pickle=False) as saved,np.load(folder/'weights.npz',allow_pickle=False) as old:
        if (saved['schema'].item()!='dgn-common11-inference-v1' or
            set(saved.files)!={'schema',*('weight_'+str(i) for i in range(22))}):
            raise ValueError('qualified complete native inference export required')
        for i in range(22):
            value=saved['weight_'+str(i)];prior=old['weight_'+str(i)]
            if value.shape!=prior.shape or value.dtype!=prior.dtype or not np.isfinite(value).all():
                raise ValueError('DGN replacement shape/dtype/finite values differ')
            digest.update(str(value.shape).encode());digest.update(str(value.dtype).encode());digest.update(value.tobytes())
    original=inventory(folder);weight_sha=sha(weights)
    shutil.copytree(folder,output);(output/'weights.npz').write_bytes(weights.read_bytes())
    if sha(output/'weights.npz')!=weight_sha or inventory(folder)!=original:
        raise ValueError('source/weights changed during replacement capture')
    metadata['model_sha256']=digest.hexdigest()
    metadata['replacement']=dict(previous_capsule_manifest_sha256=sha(folder/'artifact_hashes.json'),
        previous_model_sha256=check(folder)['model_sha256'],checkpoint_sha256=weight_sha)
    write(output/'metadata.json',metadata);write(output/'artifact_hashes.json',inventory(output))
    return metadata
