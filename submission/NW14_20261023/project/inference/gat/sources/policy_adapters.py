"""Frozen inference adapters; checkpoint training/protocol validity is a separate gate."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from candidate_sources.marl_gnn import GAT_Q_Network
from candidate_sources import raes
from candidate_sources.controls import control
from policy_timing import measure


def model_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        require_finite=torch.isfinite(value).all()
        if not require_finite:raise ValueError('nonfinite policy parameters')
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checked(observation):
    n,k=15,4
    try:
        state=np.asarray(observation['state']);current=np.asarray(observation['current'])
        adj=np.asarray(observation['adj']);ahat=np.asarray(observation['ahat']);ids=np.asarray(observation['ap_ids'])
    except (KeyError,TypeError,ValueError) as exc:raise ValueError('complete common observation required') from exc
    if state.shape!=(n,11) or state.dtype.kind not in 'iuf' or not np.isfinite(state).all():
        raise ValueError('finite fifteen-agent eleven-feature observation required')
    if current.shape!=(n,) or current.dtype.kind not in 'iu' or np.any(current<0) or np.any(current>=k):
        raise ValueError('complete actual channel indices required')
    if ids.shape!=(n,) or ids.dtype.kind not in 'iu' or len(set(ids.tolist()))!=n:
        raise ValueError('distinct stable AP identities required')
    if (adj.shape!=(n,n) or adj.dtype.kind not in 'biuf' or not np.isfinite(adj).all() or
        np.any((adj!=0)&(adj!=1)) or np.any(np.diag(adj)!=0)):
        raise ValueError('finite reported binary graph without self edges required')
    if (ahat.shape!=(n,n) or ahat.dtype.kind not in 'iuf' or not np.isfinite(ahat).all() or
        np.any(ahat<0) or np.any(np.diag(ahat)!=0)):
        raise ValueError('finite nonnegative reported receiver weights required')
    return state,current,adj,ahat,ids


def tensors(observation):
    state,_,adj,_,_=checked(observation)
    x=torch.tensor(state,dtype=torch.float32);a=torch.tensor(adj,dtype=torch.float32)
    if not torch.isfinite(x).all():raise ValueError('common features exceed float32 policy range')
    return x,a


def dgn_masks(observation):
    _,_,_,ahat,ids=checked(observation);n=len(ids)
    masks=np.zeros((n,4,n),dtype=np.float32)
    for i in range(n):
        neighbors=sorted((j for j in range(n) if j!=i),key=lambda j:(-ahat[i,j],int(ids[j])))[:3]
        for slot,j in enumerate([i]+neighbors):masks[i,slot,j]=1
    return masks


class Diagnostic:
    """Untrained controls; configuration identity remains fixed as RNG advances."""
    def __init__(self,*,method,initial_channels,seed,pref):
        elapsed=time.perf_counter_ns();cpu=time.process_time_ns()
        initial=np.asarray(initial_channels)
        if (method not in ('FIXED_INITIAL','RANDOM','PRESSURE_ONLY_M0') or initial.shape!=(15,)
            or initial.dtype.kind not in 'iu' or np.any((initial<0)|(initial>=4))
            or type(seed) is not int or seed<0 or not np.isfinite(pref) or pref<=0):
            raise ValueError('qualified diagnostic configuration required')
        self.config=dict(method=method,initial_channels=initial.tolist(),seed=seed,pref=float(pref))
        self.identity=hashlib.sha256(json.dumps(self.config,sort_keys=True,allow_nan=False).encode()).hexdigest()
        self.rng=np.random.Generator(np.random.PCG64(seed))
        self.setup_elapsed_ns=time.perf_counter_ns()-elapsed;self.setup_cpu_ns=time.process_time_ns()-cpu

    def decide(self,observation):
        (_,current,_,ahat,ids),input_cost=measure(lambda:checked(observation))
        def plan():
            if self.config['method']=='FIXED_INITIAL':return np.asarray(self.config['initial_channels'],dtype=np.int64)
            if self.config['method']=='RANDOM':return self.rng.integers(0,4,15)
            return control(np.zeros((15,4)),current,ahat,ids,method='M0',pref=self.config['pref'],mu=0,delta=0)
        action,planning_cost=measure(plan)
        return action,dict(method=self.config['method'],model_sha256=self.identity,diagnostic=True,
            phase_costs=dict(input=input_cost,planning=planning_cost))


class GATRAES:
    def __init__(self,model,*,pref,eta,lam,mu,delta,executor='incremental',batches=1):
        setup_elapsed=time.perf_counter_ns();setup_cpu=time.process_time_ns()
        if executor not in ('incremental','stale','batch','flow_guard') or type(batches) is not int or not 1<=batches<=15:
            raise ValueError('named RAES executor and one to fifteen batches required')
        if executor!='batch' and batches!=1:raise ValueError('only batch executor accepts multiple batches')
        self.executor=executor;self.batches=batches
        if type(model).__name__!='GAT_Q_Network' or model.gat.W.shape!=(4,11,16) or model.fc_policy.out_features!=4 or model.edge_dim!=0:
            raise ValueError('current shared11/64/4 GAT architecture required')
        # Verified checkpoint IO imports a fresh source module/class. Reconstruct
        # the pinned backbone from its exact state, rather than comparing class identity.
        with torch.random.fork_rng(devices=[]):
            self.model=GAT_Q_Network(11,64,4,num_heads=4,temperature=float(model.gat.temperature),
                                      dropout=model.gat.attn_dropout.p,edge_dim=0)
        try:self.model.load_state_dict(model.state_dict(),strict=True)
        except RuntimeError as exc:raise ValueError('checkpoint does not match pinned GAT backbone') from exc
        self.model.cpu().float().eval()
        for value in self.model.parameters():value.requires_grad_(False)
        self.identity=model_hash(self.model);self.coefficients=dict(pref=pref,eta=eta,lam=lam,mu=mu,delta=delta)
        # The executor validates scalar domains at construction, before first inference.
        raes.plan(np.zeros((15,4)),np.zeros(15,dtype=int),np.zeros((15,15)),np.arange(15),**self.coefficients)
        with torch.no_grad():self.model(torch.zeros(15,11),torch.zeros(15,15))
        self.setup_elapsed_ns=time.perf_counter_ns()-setup_elapsed;self.setup_cpu_ns=time.process_time_ns()-setup_cpu

    @torch.no_grad()
    def decide(self,observation):
        prepared,input_cost=measure(lambda:(tensors(observation),checked(observation)))
        (x,adj),(_,current,_,ahat,ids)=prepared
        q,neural_cost=measure(lambda:self.model(x,adj).numpy())
        def validate():
            if q.shape!=(15,4) or not np.isfinite(q).all() or model_hash(self.model)!=self.identity:
                raise ValueError('invalid/changed GAT Q or weights')
        _,validation_cost=measure(validate)
        def planning():
            if self.executor in ('incremental','flow_guard'):
                action,trace=raes.plan(q,current,ahat,ids,capture_trace=True,**self.coefficients)
                return action,[int(row['node']) for row in trace]
            action=control(q,current,ahat,ids,method='X2' if self.executor=='stale' else 'X3',
                batches=self.batches,**self.coefficients)
            pressure=ahat@np.eye(4)[current]
            order=sorted(range(15),key=lambda i:(-pressure[i,current[i]],int(ids[i])))
            return action,order
        (action,order),planning_cost=measure(planning)
        evidence=dict(method='GAT+RAES',model_sha256=self.identity,q=q.tolist(),
            coefficients=self.coefficients,reservation_order=order,executor=self.executor,batches=self.batches,
            phase_costs=dict(input=input_cost,neural=neural_cost,validation=validation_cost,pressure_and_planning=planning_cost))
        if self.executor=='flow_guard':
            def guard():
                m0=control(np.zeros_like(q),current,ahat,ids,method='M0',pref=self.coefficients['pref'])
                pressure=lambda channels:np.sum(ahat*(channels[:,None]==channels[None,:]),axis=1)
                learned_pressure=pressure(action);m0_pressure=pressure(m0)
                worse=learned_pressure>m0_pressure+1e-12
                fallback=bool(np.any(worse));selected=m0 if fallback else action
                return selected,dict(fallback=fallback,reason='FLOW_PRESSURE_ABOVE_M0' if fallback else 'NO_FLOW_PRESSURE_INCREASE',
                    tolerance=1e-12,learned_channels=action.tolist(),m0_channels=m0.tolist(),
                    learned_pressure=learned_pressure.tolist(),m0_pressure=m0_pressure.tolist(),
                    worse_flow_count=int(worse.sum()),action_m0_agreement=float(np.mean(selected==m0)),
                    learned_distinct_node_choices=int(np.sum(selected!=m0)))
            (action,detail),cost=measure(guard)
            evidence.update(method='GAT_M0_FLOW_GUARD',guard=detail)
            evidence['phase_costs']['m0_and_flow_guard']=cost
        return action,evidence


class GANet:
    def __init__(self,learner):
        setup_elapsed=time.perf_counter_ns();setup_cpu=time.process_time_ns()
        if learner.config['agents']!=15 or learner.config['observation_dim']!=11 or learner.config['actions']!=4:
            raise ValueError('shared15/11/4 GA-Net checkpoint required')
        self.model=deepcopy(learner.target_actors).cpu().float().eval()
        for value in self.model.parameters():value.requires_grad_(False)
        self.identity=model_hash(self.model)
        with torch.no_grad():self.model(torch.zeros(1,15,11))
        self.setup_elapsed_ns=time.perf_counter_ns()-setup_elapsed;self.setup_cpu_ns=time.process_time_ns()-setup_cpu

    @torch.no_grad()
    def decide(self,observation):
        (x,_),input_cost=measure(lambda:tensors(observation))
        p,neural_cost=measure(lambda:self.model(x.unsqueeze(0))[0].numpy())
        def validate():
            if p.shape!=(15,4) or not np.isfinite(p).all() or not np.allclose(p.sum(-1),1) or model_hash(self.model)!=self.identity:
                raise ValueError('invalid/changed local actor probabilities or weights')
        _,validation_cost=measure(validate)
        action,action_cost=measure(lambda:p.argmax(-1))
        return action,dict(method='GA-Net',model_sha256=self.identity,
            policy='target_local_categorical_argmax',probabilities=p.tolist(),
            phase_costs=dict(input=input_cost,neural=neural_cost,validation=validation_cost,action_selection=action_cost))


class DGN:
    def __init__(self,*,sources=None,weights=None,seed=None,output,capsule=None,owner_root=None):
        from live_runtime import ProcessLines
        import policy_capsule
        setup_elapsed=time.perf_counter_ns()
        here=Path(__file__).resolve().parent;output=Path(output).resolve()
        owner=here if owner_root is None else Path(owner_root).resolve()
        if owner not in output.parents or output.exists():raise ValueError('fresh DGN worker evidence inside comparison root required')
        runtime=here.parent/'dgn_cpu_preparation_20261001/user_routing_setup/20261002_172101_785a71f36c4b49b79a651593a693f599/venv/Scripts/python.exe'
        output.mkdir(parents=True)
        if capsule is None:
            self.capsule_path=output/'capsule';meta=policy_capsule.snapshot_dgn(sources,weights,seed,self.capsule_path)
        else:
            self.capsule_path=Path(capsule).resolve();meta=policy_capsule.check(self.capsule_path)
            runtime=Path(meta['runtime']['executable']);seed=meta['seed']
        command=[str(runtime),'-I','-B','-u',str(self.capsule_path/'sources/dgn_worker.py'),
            '--sources',str(self.capsule_path/'graphs'),'--seed',str(seed),'--weights',str(self.capsule_path/'weights.npz'),
            '--stage-path',str(self.capsule_path/'sources/historical/dgn_routing_execution_20261003/model_stage.py')]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
        self._child=ProcessLines(command,output/'stdout.log',env=env)
        try:
            ready=self._response('DGN_READY ')
            if ready['feature_width']!=11 or ready['aggregation']!='UPSTREAM_RAW_SCORES' or ready['optimizer_steps']!=0:
                raise ValueError('native primary DGN identity differs')
            self.identity=ready['model_sha256'];self.setup_cpu_ns=ready['setup_and_zero_fixture_warmup_cpu_ns']
            if self.identity!=meta['model_sha256']:raise ValueError('captured DGN weights differ from actual native model')
            if capsule is None:policy_capsule.bind_runtime(self.capsule_path,ready)
            self.setup_elapsed_ns=time.perf_counter_ns()-setup_elapsed
            self.output=output
            self.owner_root=owner
            (output/'worker_identity.json').write_text(json.dumps(dict(ready=ready,
                capsule_manifest_sha256=policy_capsule.sha(self.capsule_path/'artifact_hashes.json'),
                checkpoint_sha256=policy_capsule.sha(self.capsule_path/'weights.npz'),
                worker_source_sha256=policy_capsule.sha(self.capsule_path/'sources/dgn_worker.py'),
                source_manifest_sha256=policy_capsule.sha(self.capsule_path/'graphs/source_manifest.json'),
                command=command),indent=2),encoding='utf-8')
        except Exception:
            self._child.close();raise

    def _response(self,prefix):
        deadline=time.monotonic()+180
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('native DGN response deadline')
            line=self._child.next(remaining)
            if line is None:raise RuntimeError('native DGN worker ended before response')
            if line.startswith(prefix):return json.loads(line[len(prefix):])

    def decide(self,observation):
        (state,masks),input_cost=measure(lambda:(checked(observation)[0],dgn_masks(observation)))
        def rpc():
            self._child.send(json.dumps(dict(op='predict',state=state.tolist(),masks=masks.tolist()),allow_nan=False))
            return self._response('DGN_RESPONSE ')
        response,rpc_cost=measure(rpc)
        def validate_action():
            q=np.asarray(response['q'])
            if q.shape!=(15,4) or not np.isfinite(q).all() or response['model_sha256']!=self.identity:
                raise ValueError('invalid/changed frozen native DGN prediction')
            return q,q.argmax(-1)
        (q,action),validation_cost=measure(validate_action)
        return action,dict(method='DGN',model_sha256=self.identity,q=q.tolist(),
            primary_variant='RAW_PLUS_KL_LOCAL_COMMON11',neighbor_indices=masks.argmax(-1).tolist(),
            external_cpu_ns=response['cpu_ns'],worker_phase_costs=response['phase_costs'],
            phase_costs=dict(input_and_masks=input_cost,rpc=rpc_cost,validation_and_action=validation_cost))

    def refresh(self,*,weights,output):
        import policy_capsule
        output=Path(output).resolve()
        if self._child is None:raise ValueError('native inference worker is closed')
        if self.owner_root not in output.parents or output.exists():raise ValueError('fresh owned DGN refresh evidence required')
        elapsed=time.perf_counter_ns();cpu=time.process_time_ns()
        metadata=policy_capsule.refresh_dgn(self.capsule_path,weights,output/'capsule')
        request=dict(op='refresh',path=str(output/'capsule/weights.npz'),previous_model_sha256=self.identity,
            model_sha256=metadata['model_sha256'],checkpoint_sha256=policy_capsule.sha(output/'capsule/weights.npz'))
        self._child.send(json.dumps(request,allow_nan=False));response=self._response('DGN_RESPONSE ')
        if (response['op']!='refresh' or response['previous_model_sha256']!=self.identity or
            response['model_sha256']!=metadata['model_sha256'] or response['checkpoint_sha256']!=request['checkpoint_sha256']
            or response['optimizer_steps']!=0):raise ValueError('actual DGN replacement differs from capsule')
        self.identity=response['model_sha256'];self.capsule_path=output/'capsule'
        self.setup_cpu_ns=time.process_time_ns()-cpu+response['cpu_ns'];self.setup_elapsed_ns=time.perf_counter_ns()-elapsed
        result=dict(response,elapsed_ns=self.setup_elapsed_ns,parent_and_worker_cpu_ns=self.setup_cpu_ns,
            worker_pid=self._child.process.pid,request=request,
            capsule_manifest_sha256=policy_capsule.sha(self.capsule_path/'artifact_hashes.json'))
        policy_capsule.write(output/'refresh.json',result)
        policy_capsule.write(output/'artifact_hashes.json',policy_capsule.inventory(output))
        return result

    def close(self):
        if self._child is not None:
            self._child.send(json.dumps(dict(op='exit')));code=self._child.close();self._child=None
            (self.output/'terminal.json').write_text(json.dumps(dict(status='TERMINAL',exit_code=code)),encoding='utf-8')
            if code:raise RuntimeError('native DGN worker failed at shutdown')

    def __enter__(self):return self
    def __exit__(self,*exception):self.close()
