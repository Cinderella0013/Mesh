"""Collect a frozen own-policy packet episode, then update and save its learner."""
import json
from pathlib import Path
import time
import numpy as np
import torch
import audit_live
from behavior_policy import BehaviorPolicy
import campaign_inputs
import live_runtime
import physical_replay
from policy_adapters import GATRAES,GANet,DGN,model_hash
HERE=Path(__file__).resolve().parent


def serializable(value):
    if isinstance(value,(torch.Tensor,np.ndarray)):return value.tolist()
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serializable(v) for v in value]
    return value


def run(method,learner,*,reference,arguments,tokens,output,behavior_seed,epsilon,coefficients,scope,inference=None,resource_profile=None):
    output=Path(output).resolve()
    if HERE not in output.parents or output.exists():raise ValueError('fresh owned training episode output required')
    if method not in ('GAT+RAES','GA-Net','DGN_RAW_KL_LOCAL'):raise ValueError('qualified episode method required')
    if not isinstance(scope,str) or not scope:raise ValueError('explicit experiment scope required')
    if inference is not None and (method!='DGN_RAW_KL_LOCAL' or type(inference).__name__!='DGN'):
        raise ValueError('persistent inference is available only for the qualified native DGN policy')
    output.mkdir(parents=True);start=time.perf_counter_ns();cpu=time.process_time_ns()
    native=method=='DGN_RAW_KL_LOCAL'
    count=lambda:learner.request(op='info')['td_steps'] if native else learner.updates
    before=count()
    receipt=dict(status='STARTED',scope=scope,method=method,updates_before=before)
    campaign_inputs.write(output/'receipt.json',receipt)
    paths=list(HERE.glob('*.py'))+list((HERE/'candidate_sources').glob('*.py'))
    source_hashes={str(p.relative_to(HERE)):campaign_inputs.sha(p) for p in paths}
    for path in paths:
        copied=output/'sources'/path.relative_to(HERE);copied.parent.mkdir(parents=True,exist_ok=True)
        copied.write_bytes(path.read_bytes())
    campaign_inputs.write(output/'source_hashes.json',source_hashes)
    base=None
    try:
        if native:
            exported=learner.output/('collector_'+str(before)+'.npz')
            identity=learner.request(op='export',path=str(exported))['model_sha256']
            initialization_seed=json.loads((learner.output/'receipt.json').read_text())['seed']
            if inference is None:
                base=DGN(sources=learner.output/'graphs',weights=exported,seed=initialization_seed,output=output/'inference_worker')
            else:
                import policy_capsule
                base=inference;metadata=policy_capsule.check(base.capsule_path)
                if (metadata['seed']!=initialization_seed or
                    campaign_inputs.sha(base.capsule_path/'graphs/source_manifest.json')!=campaign_inputs.sha(learner.output/'graphs/source_manifest.json')
                    or campaign_inputs.sha(base.capsule_path/'sources/dgn_worker.py')!=campaign_inputs.sha(learner.output/'dgn_worker.py')):
                    raise ValueError('persistent collector seed/graph/builder differs from actual native learner')
                if base.identity!=identity:base.refresh(weights=exported,output=output/'inference_refresh')
                receipt['persistent_inference_worker_pid']=base._child.process.pid
            if base.identity!=identity:raise ValueError('collector differs from actual native learner export')
        else:base=GATRAES(learner.online,**coefficients) if method=='GAT+RAES' else GANet(learner)
        policy=BehaviorPolicy(base,rule='categorical' if method=='GA-Net' else 'epsilon_greedy',
            seed=behavior_seed,epsilon=0 if method=='GA-Net' else epsilon)
        collector=policy.identity
        live=live_runtime.run(reference,output/'collection',arguments,policy,tokens=tokens,resource_profile=resource_profile)
        if live['exit_code']!=0 or live['decisions']!=13 or count()!=before:
            raise ValueError('complete frozen thirteen-decision own-policy episode required')
        corpus=physical_replay.collect(output/'collection',tokens=tokens)
        decisions=[json.loads(line) for line in (output/'collection/decisions.jsonl').read_text().splitlines()]
        if len(corpus['transitions'])!=13:raise ValueError('complete thirteen-transition learner corpus required')
        for transition,decision in zip(corpus['transitions'],decisions):
            if decision['policy_evidence']['model_sha256']!=collector or decision['channels']!=transition['intended_actions'].tolist():
                raise ValueError('training corpus differs from frozen collector or actual attempted action')
        # Audit before the first update; failed native collection never silently
        # becomes a shortened or substituted training episode.
        audit=audit_live.audit(output/'collection')
        campaign_inputs.write(output/'collection_audit.json',audit)
        if native and inference is None:base.close()
        learning_start=time.perf_counter_ns();learning_cpu=time.process_time_ns()
        with (output/'updates.jsonl').open('x',encoding='utf-8') as stream:
            for index,transition in enumerate(corpus['transitions']):
                if native:result=learner.request(op='train',transition=physical_replay.dgn_payload(transition))
                else:result=learner.train_transition(transition) if method=='GAT+RAES' else learner.train_batch(**physical_replay.ganet_batch(corpus,[index]))
                stream.write(json.dumps(dict(transition_index=index,intended_actions=transition['intended_actions'].tolist(),
                    result=serializable(result)),allow_nan=False)+'\n');stream.flush()
        learning_elapsed=time.perf_counter_ns()-learning_start;learning_cpu=time.process_time_ns()-learning_cpu
        after=count()
        if after!=before+13:raise ValueError('learner exposure differs from retained episode transitions')
        if any(campaign_inputs.sha(HERE/name)!=digest for name,digest in source_hashes.items()):
            raise ValueError('source changed during episode collection/training')
        checkpoint=output/('checkpoint.npz' if native else 'checkpoint.pt')
        provenance=dict(scope=scope,collector_model_sha256=collector,
            corpus_manifest_sha256=corpus['source_manifest_sha256'],episode_source_hashes=source_hashes)
        if native:
            owned_checkpoint=learner.output/('checkpoint_'+str(after)+'.npz')
            learner.request(op='save',path=str(owned_checkpoint))
            checkpoint.write_bytes(owned_checkpoint.read_bytes())
            campaign_inputs.write(output/'checkpoint_reference.json',dict(actual_saved_path=str(owned_checkpoint),
                checkpoint_sha256=campaign_inputs.sha(owned_checkpoint),provenance=provenance))
        elif method=='GAT+RAES':learner.save(checkpoint,provenance=provenance)
        else:learner.save(checkpoint)
        receipt.update(status='PASS',updates_after=after,collector_model_sha256=collector,
            next_model_sha256=learner.request(op='info')['online_sha256'] if native else model_hash(learner.online if method=='GAT+RAES' else learner.target_actors),
            checkpoint=str(checkpoint),checkpoint_sha256=campaign_inputs.sha(checkpoint),
            corpus_manifest_sha256=corpus['source_manifest_sha256'],behavior=policy.checkpoint(),
            collection_elapsed_seconds=live['elapsed_seconds'],learning_elapsed_ns=learning_elapsed,
            learning_parent_cpu_ns=learning_cpu,elapsed_ns=time.perf_counter_ns()-start,
            parent_cpu_ns=time.process_time_ns()-cpu,source_hashes=source_hashes)
    except Exception as exc:
        receipt.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),updates_after=None if native else learner.updates)
        raise
    finally:
        if native and base is not None and inference is None:base.close()
        campaign_inputs.write(output/'receipt.json',receipt)
        campaign_inputs.write(output/'artifact_hashes.json',campaign_inputs.inventory(output))
    return receipt
