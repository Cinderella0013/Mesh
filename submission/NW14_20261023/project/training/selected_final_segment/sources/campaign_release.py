"""Immutable prospective protocol; draft inputs never authorize a scientific job."""
import json
from pathlib import Path
import sys
import campaign_inputs
import campaign_jobs
import live_runtime
import raes_calibration
HERE=Path(__file__).resolve().parent
ROLES={'integrated_resources':'ACTUAL_RESOURCE_ENGINEERING_ONLY',
       'native_workers':'ACTUAL_TF_WORKER_RESOURCE_ENGINEERING_ONLY',
       'seed_loop':'SEGMENTED_SEED_ENGINEERING_ONLY',
       'metric_consumer':'METRIC_CONSUMER_ENGINEERING_ONLY',
       'static_dispatch':'ALLTHREE_STATIC_AND_DIAGNOSTIC_ENGINEERING_ONLY',
       'static_full_cost':'STATIC_METRIC_FULL_COST_ENGINEERING_ONLY',
       'dynamic_diagnostics':'DYNAMIC_DIAGNOSTIC_ENGINEERING_ONLY',
       'shared_store':'SHARED_TIER_STORAGE_ENGINEERING_ONLY',
       'staging':'RECOVERABLE_NEW_STAGING_ENGINEERING_ONLY',
       'seed_archive':'RECOVERABLE_SEED_CONSUMER_ENGINEERING_ONLY',
       'static_reuse':'EXACT_STATIC_PAYLOAD_REUSE_ENGINEERING_ONLY',
       'execution_basis':'GRAPH_RUNTIME_BASIS_ENGINEERING_ONLY',
       'main_consumers':'RELEASED_CONTEXT_CONSUMER_ENGINEERING_ONLY'}


def production():
    paths=list(HERE.glob('*.py'))+list((HERE/'candidate_sources').glob('*.py'))
    paths += [HERE/name for name in ('METHOD_SPEC.md','PHYSICAL_CONTRACT.md','PROTOCOL_RULES_DRAFT.md')]
    paths += list((HERE/'native').glob('*.cc'))
    return {path.relative_to(HERE).as_posix():campaign_inputs.sha(path) for path in paths if not path.name.startswith('test_')}


def closed(folder):
    folder=Path(folder).resolve()
    if campaign_inputs.inventory(folder)!=json.loads((folder/'artifact_hashes.json').read_text()):
        raise ValueError('qualification closed inventory changed')
    record=json.loads((folder/'receipt.json').read_text())
    if record['status']!='PASS':raise ValueError('successful complete qualification required')
    return record


def execution_basis(graphs,native_proof):
    import numpy as np
    import torch
    graphs=Path(graphs).resolve();native_proof=Path(native_proof).resolve()
    if closed(native_proof)['scope']!=ROLES['native_workers']:
        raise ValueError('qualified actual native worker runtime required')
    ready=json.loads((native_proof/'learner/ready.json').read_text())
    graph_hashes=campaign_inputs.inventory(graphs)
    if (graph_hashes!=campaign_inputs.inventory(native_proof/'learner/graphs')
        or campaign_inputs.sha(graphs/'source_manifest.json')!=ready['basis']['graphs']):
        raise ValueError('actual qualified native graph identity differs')
    native_files={name:digest for name,digest in ready['runtime_dependencies'].items()
        if HERE not in Path(name).resolve().parents}
    runtime=ready['runtime']
    native_files.update({runtime['executable']:runtime['executable_sha256'],
        runtime['base_executable']:runtime['base_executable_sha256'],
        str(Path(runtime['executable']).parent.parent/'pyvenv.cfg'):runtime['venv_config_sha256']})
    historical=['dgn_routing_execution_20261003/model_stage.py']+[
        'dgn_cpu_preparation_20261001/'+name for name in ('routing_target_probe.py','routing_invariants_probe.py',
            'routing_full_model_probe.py','routing_repair_probe.py','routing_td_probe.py','routing_probe.py')]
    historical_files={str((HERE.parent/name).resolve()):campaign_inputs.sha(HERE.parent/name) for name in historical}
    modern_files={str(Path(module.__file__).resolve()):campaign_inputs.sha(module.__file__)
        for module in list(sys.modules.values()) if getattr(module,'__file__',None)
        and Path(module.__file__).is_file() and HERE not in Path(module.__file__).resolve().parents}
    modern_files.update({str(Path(sys.executable).resolve()):campaign_inputs.sha(sys.executable),
        str(Path(sys._base_executable).resolve()):campaign_inputs.sha(sys._base_executable)})
    basis=dict(graphs=str(graphs),graph_hashes=graph_hashes,native_runtime=runtime,
        native_installed_files=native_files,historical_sources=historical_files,modern_installed_files=modern_files,
        modern_runtime=dict(executable=str(Path(sys.executable).resolve()),python=sys.version,torch=torch.__version__,numpy=np.__version__))
    check_execution_basis(basis)
    return basis


def check_execution_basis(basis):
    import numpy as np
    import torch
    if campaign_inputs.inventory(Path(basis['graphs']))!=basis['graph_hashes']:
        raise ValueError('released graph bytes changed')
    actual=dict(executable=str(Path(sys.executable).resolve()),python=sys.version,torch=torch.__version__,numpy=np.__version__)
    if actual!=basis['modern_runtime']:raise ValueError('released modern runtime identity differs')
    for group in ('native_installed_files','historical_sources','modern_installed_files'):
        for name,digest in basis[group].items():
            if campaign_inputs.sha(name)!=digest:raise ValueError('released execution dependency changed: '+name)
    return basis


def freeze(*,inputs,calibration,reference,qualifications,output,graphs=None):
    inputs=Path(inputs).resolve();calibration=Path(calibration).resolve();reference=Path(reference).resolve()
    output=Path(output).resolve()
    if set(qualifications)!=set(ROLES):raise ValueError('complete qualification role set required before protocol release')
    proofs={}
    for role,path in qualifications.items():
        path=Path(path).resolve();record=closed(path)
        if record['scope']!=ROLES[role]:raise ValueError('qualification scope differs: '+role)
        proofs[role]=dict(path=str(path),manifest_sha256=campaign_inputs.sha(path/'artifact_hashes.json'))
    campaign_inputs.verify(inputs);plan=json.loads((inputs/'plan.json').read_text())
    measured=closed(calibration)
    records=json.loads((calibration/'measurements.json').read_text())
    if (measured['scope']!='TRAINING_ONLY_PRETRAFFIC_CALIBRATION'
        or [r['topology_seed'] for r in records]!=plan['topology_seeds']['train']
        or [r['packet_seed'] for r in records]!=list(range(9719000,9719030))
        or len(records)!=30 or any(r['split']!='train' or r['pilot_available_at_seconds']>=1 for r in records)):
        raise ValueError('complete thirty training-only calibration records required')
    coefficients=raes_calibration.coefficients(records,training_topologies=plan['topology_seeds']['train'])
    if coefficients!=measured['coefficients']:raise ValueError('frozen calibration coefficients differ')
    reference_record,_,_,binary_validation=live_runtime.verified_binary(reference)
    if reference_record.get('native_resources',{}).get('cpus')!=[2]:raise ValueError('qualified actual native CPU reference required')
    sources=production()
    if not all(name in sources for name in ('main_campaign.py','main_evaluation.py')):
        raise ValueError('qualified actual main training/evaluation consumers required')
    if graphs is None:raise ValueError('qualified actual native graph path required')
    basis=execution_basis(graphs,qualifications['native_workers'])
    if HERE not in output.parents or output.exists():raise ValueError('fresh owned protocol output required')
    jobs=campaign_jobs.build(plan)
    if len(jobs['training'])!=3000 or len(jobs['evaluation'])!=24768:
        raise ValueError('full original prospective job matrix required')
    # All prerequisites are checked before any release folder is written.
    output.mkdir(parents=True)
    for name in sources:
        target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((HERE/name).read_bytes())
    campaign_inputs.write(output/'plan.json',plan);campaign_inputs.write(output/'jobs.json',jobs)
    campaign_inputs.write(output/'coefficients.json',coefficients)
    record=dict(schema='threeway-released-protocol-v1',status='RELEASED',
        inputs=str(inputs),input_manifest_sha256=campaign_inputs.sha(inputs/'artifact_hashes.json'),
        calibration=str(calibration),calibration_manifest_sha256=campaign_inputs.sha(calibration/'artifact_hashes.json'),
        reference=str(reference),reference_manifest_sha256=campaign_inputs.sha(reference/'artifact_hashes.json'),
        graphs=basis['graphs'],execution_basis=basis,
        binary_sha256=reference_record['binary_sha256'],binary_validation=binary_validation,
        qualifications=proofs,source_hashes=sources,jobs_sha256=campaign_inputs.sha(output/'jobs.json'),
        plan_sha256=campaign_inputs.sha(output/'plan.json'),coefficients_sha256=campaign_inputs.sha(output/'coefficients.json'),
        independent_models_per_method=10,episodes_per_model=100,static_and_dynamic_required=True,
        staging_segment_episodes=10,staging_rule='CLOSED_SEGMENTS_ONLY; EXACT_RESTORE_BEFORE_PREFIX_RESUME',
        setup_training_archive_cost='REPORTED_SEPARATELY_FROM_CHARGED_POLICY_DECISIONS')
    campaign_inputs.write(output/'protocol.json',record)
    campaign_inputs.write(output/'artifact_hashes.json',campaign_inputs.inventory(output))
    verify(output)
    return record


def verify(folder):
    folder=Path(folder).resolve()
    if not folder.is_dir():raise ValueError('actual released protocol directory required')
    record=json.loads((folder/'protocol.json').read_text())
    if (record.get('schema')!='threeway-released-protocol-v1' or record.get('status')!='RELEASED'
        or record.get('independent_models_per_method')!=10 or record.get('episodes_per_model')!=100
        or record.get('static_and_dynamic_required') is not True):raise ValueError('full scientifically released protocol required')
    if campaign_inputs.inventory(folder)!=json.loads((folder/'artifact_hashes.json').read_text()):
        raise ValueError('released protocol inventory changed')
    if production()!=record['source_hashes']:raise ValueError('current executed sources differ from released protocol')
    check_execution_basis(record['execution_basis'])
    for name,digest in record['source_hashes'].items():
        if campaign_inputs.sha(folder/'sources'/name)!=digest:raise ValueError('frozen protocol source changed')
    for filename,key in (('jobs.json','jobs_sha256'),('plan.json','plan_sha256'),('coefficients.json','coefficients_sha256')):
        if campaign_inputs.sha(folder/filename)!=record[key]:raise ValueError('released '+filename+' changed')
    inputs=Path(record['inputs']);campaign_inputs.verify(inputs)
    if campaign_inputs.sha(inputs/'artifact_hashes.json')!=record['input_manifest_sha256']:
        raise ValueError('released input identity changed')
    plan=json.loads((folder/'plan.json').read_text());jobs=json.loads((folder/'jobs.json').read_text())
    if jobs!=campaign_jobs.build(plan):raise ValueError('released complete deterministic job matrix differs')
    reference=Path(record['reference']);actual,_,_,_=live_runtime.verified_binary(reference)
    if (campaign_inputs.sha(reference/'artifact_hashes.json')!=record['reference_manifest_sha256']
        or actual['binary_sha256']!=record['binary_sha256']):raise ValueError('released native reference changed')
    calibration=Path(record['calibration']);measured=closed(calibration)
    if campaign_inputs.sha(calibration/'artifact_hashes.json')!=record['calibration_manifest_sha256']:
        raise ValueError('released calibration identity changed')
    measurements=json.loads((calibration/'measurements.json').read_text())
    if (len(measurements)!=30 or [r['topology_seed'] for r in measurements]!=plan['topology_seeds']['train']
        or [r['packet_seed'] for r in measurements]!=list(range(9719000,9719030))
        or measured['scope']!='TRAINING_ONLY_PRETRAFFIC_CALIBRATION'):
        raise ValueError('released training-only calibration coverage differs')
    coefficients=raes_calibration.coefficients(measurements,training_topologies=plan['topology_seeds']['train'])
    if coefficients!=json.loads((folder/'coefficients.json').read_text()) or coefficients!=measured['coefficients']:
        raise ValueError('released measured coefficients differ')
    if set(record['qualifications'])!=set(ROLES):raise ValueError('released qualification roles incomplete')
    for role,proof in record['qualifications'].items():
        path=Path(proof['path']);qualification=closed(path)
        if qualification['scope']!=ROLES[role] or campaign_inputs.sha(path/'artifact_hashes.json')!=proof['manifest_sha256']:
            raise ValueError('released qualification changed: '+role)
    return record


def seed_configuration(protocol,method,seed):
    protocol=Path(protocol).resolve();record=verify(protocol)
    jobs=json.loads((protocol/'jobs.json').read_text())['training']
    own=[job for job in jobs if job['method']==method and job['training_seed']==seed]
    if len(own)!=100 or [job['episode'] for job in own]!=list(range(100)):
        raise ValueError('complete declared own-method independent training seed required')
    return dict(schema='threeway-seed-training-v1',scope='MAIN_RELEASED_PROTOCOL',method=method,seed=seed,jobs=own,
        protocol=str(protocol),protocol_manifest_sha256=campaign_inputs.sha(protocol/'artifact_hashes.json'),
        inputs=record['inputs'],reference=record['reference'],graphs=record['graphs'],
        coefficients=json.loads((protocol/'coefficients.json').read_text()))
