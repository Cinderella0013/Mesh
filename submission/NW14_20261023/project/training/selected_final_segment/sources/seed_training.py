"""Serial own-policy seed training, with closed-prefix checkpoint continuation.

The engineering CLI qualifies this consumer before the main protocol is released.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
HERE=Path(__file__).resolve().parent


def checked_configuration(config):
    if config.get('schema')!='threeway-seed-training-v1':
        raise ValueError('declared seed training configuration required')
    method=config.get('method');seed=config.get('seed');jobs=config.get('jobs')
    if config.get('scope')=='MAIN_RELEASED_PROTOCOL':
        import campaign_release
        if type(seed) is not int:raise ValueError('declared independent model seed required')
        expected=campaign_release.seed_configuration(config['protocol'],method,seed)
        if config!=expected:raise ValueError('seed configuration differs from complete released protocol')
        return method,seed,jobs,'MAIN_SEED_LOOP'
    if config.get('scope')!='ENGINEERING_ONLY':
        raise ValueError('engineering or actually released main configuration required')
    if method not in ('GAT+RAES','GA-Net','DGN_RAW_KL_LOCAL') or type(seed) is not int or not 9790000<=seed<=9799999:
        raise ValueError('qualified method and separate engineering initialization seed required')
    if (not isinstance(jobs,list) or not jobs or len({j.get('id') for j in jobs})!=len(jobs) or
        any(j.get('episode')!=i or j.get('training_seed')!=seed or j.get('method')!=method
            or j.get('task')!='development' for i,j in enumerate(jobs))):
        raise ValueError('complete ordered own-method engineering configuration jobs required')
    return method,seed,jobs,'ENGINEERING_SEED_LOOP_ONLY'


def run(configuration,output,*,resume=None,max_episodes=None):
    # Set BLAS/thread environment before importing NumPy or the learners.
    import campaign_resources
    resources=campaign_resources.apply()
    import campaign_inputs
    import campaign_materialize
    import torch
    from policy_adapters import DGN,model_hash
    import training_episode
    configuration=Path(configuration).resolve();output=Path(output).resolve()
    config=json.loads(configuration.read_text());config_sha=campaign_inputs.sha(configuration)
    method,seed,jobs,scope=checked_configuration(config)
    if max_episodes is not None and (type(max_episodes) is not int or max_episodes<1):
        raise ValueError('positive checkpoint segment budget required')
    if HERE not in output.parents or output.exists():raise ValueError('fresh owned seed output required')
    production=list(HERE.glob('*.py'))+list((HERE/'candidate_sources').glob('*.py'))
    production=[p for p in production if not p.name.startswith('test_')]
    sources={p.relative_to(HERE).as_posix():campaign_inputs.sha(p) for p in production}
    completed=0;checkpoint=None;previous_sha=None;previous_model=None
    if resume is not None:
        resume=Path(resume).resolve()
        if HERE not in resume.parents:raise ValueError('owned closed seed prefix required')
        old=json.loads((resume/'receipt.json').read_text())
        if campaign_inputs.inventory(resume)!=json.loads((resume/'artifact_hashes.json').read_text()):
            raise ValueError('closed seed prefix artifact inventory changed')
        if old['configuration_sha256']!=config_sha or old['method']!=method or old['seed']!=seed:
            raise ValueError('resume configuration differs from closed prefix')
        if old['source_hashes']!=sources:raise ValueError('resume executed consumer sources differ')
        completed=old['episodes_completed']
        if old['status']!='CHECKPOINTED' or not 0<completed<len(jobs) or old['updates_completed']!=13*completed:
            raise ValueError('successful unfinished closed seed prefix required')
        checkpoint=Path(old['checkpoint']).resolve()
        if resume not in checkpoint.parents or campaign_inputs.sha(checkpoint)!=old['checkpoint_sha256']:
            raise ValueError('resume actual checkpoint identity differs')
        previous_sha=campaign_inputs.sha(resume/'artifact_hashes.json');previous_model=old['model_sha256']
    inputs=Path(config['inputs']).resolve();campaign_inputs.verify(inputs)
    output.mkdir();started=time.perf_counter_ns()
    shutil.copyfile(configuration,output/'configuration.json')
    for path in production:
        target=output/'sources'/path.relative_to(HERE);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    receipt=dict(status='STARTED',scope=scope,method=method,seed=seed,
        configuration_sha256=config_sha,planned_episodes=len(jobs),episodes_completed=completed,
        updates_completed=13*completed,source_hashes=sources,resources=resources,
        resume=str(resume) if resume else None,resume_manifest_sha256=previous_sha)
    campaign_inputs.write(output/'receipt.json',receipt)
    learner=inference=None;native=method=='DGN_RAW_KL_LOCAL'
    try:
        if native:
            from dgn_learning_client import NativeLearner
            learner=NativeLearner(graphs=config['graphs'],seed=seed,output=output/'learner',resume=checkpoint)
            identity=lambda:learner.request(op='info')['online_sha256']
            count=lambda:learner.request(op='info')['td_steps']
            initial=learner.output/'seed_initial.npz';learner.request(op='export',path=str(initial))
            inference=DGN(sources=learner.output/'graphs',weights=initial,seed=seed,output=output/'inference')
        elif method=='GAT+RAES':
            from gat_learning import Learner
            learner=Learner.load(checkpoint) if checkpoint else Learner(seed=seed)
            identity=lambda:model_hash(learner.online);count=lambda:learner.updates
        else:
            from ganet import Learner
            if checkpoint:learner=Learner.load(checkpoint)
            else:torch.manual_seed(seed);learner=Learner(15,11,4)
            identity=lambda:model_hash(learner.target_actors);count=lambda:learner.updates
        if count()!=13*completed or previous_model is not None and identity()!=previous_model:
            raise ValueError('loaded seed model/update identity differs from closed prefix')
        stop=len(jobs) if max_episodes is None else min(len(jobs),completed+max_episodes)
        for index in range(completed,stop):
            job=jobs[index]
            condition=campaign_materialize.prepare(inputs,job,reference=config['reference'],output=output/'inputs'/f'{index:03}')
            expected=identity()
            kwargs=dict(reference=config['reference'],arguments=condition['arguments'],tokens=condition['tokens'],
                output=output/'episodes'/f'{index:03}',behavior_seed=job['behavior_seed'],epsilon=job['epsilon'],
                coefficients=config['coefficients'],scope=scope,resource_profile=resources)
            if native:kwargs['inference']=inference
            result=training_episode.run(method,learner,**kwargs)
            if result['collector_model_sha256']!=expected or count()!=13*(index+1):
                raise ValueError('seed episode did not use its actual current policy or complete exposure')
            checkpoint=Path(result['checkpoint'])
            receipt.update(episodes_completed=index+1,updates_completed=count(),checkpoint=str(checkpoint),
                checkpoint_sha256=campaign_inputs.sha(checkpoint),model_sha256=identity())
            campaign_inputs.write(output/'receipt.json',receipt)
            print('SEED_EPISODE_COMPLETE',method,seed,index+1,'/',len(jobs),flush=True)
        if any(campaign_inputs.sha(HERE/name)!=digest for name,digest in sources.items()):
            raise ValueError('executed seed consumer sources changed during training')
        receipt['status']='PASS' if stop==len(jobs) else 'CHECKPOINTED'
    except Exception as exc:
        receipt.update(status='FAILED',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        try:
            if inference is not None:inference.close()
            if native and learner is not None:learner.close()
        except Exception as exc:
            receipt.update(status='FAILED_SHUTDOWN',shutdown_error=repr(exc));raise
        finally:
            receipt['elapsed_ns']=time.perf_counter_ns()-started
            campaign_inputs.write(output/'receipt.json',receipt)
            campaign_inputs.write(output/'artifact_hashes.json',campaign_inputs.inventory(output))
    return receipt


if __name__=='__main__':
    sys.path.insert(0,str(HERE))
    parser=argparse.ArgumentParser();parser.add_argument('--configuration',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--resume');parser.add_argument('--max-episodes',type=int)
    args=parser.parse_args()
    result=run(args.configuration,args.output,resume=args.resume,max_episodes=args.max_episodes)
    print('SEED_TRAINING',result['status'],result['episodes_completed'],result['updates_completed'],flush=True)
