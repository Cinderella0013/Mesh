"""Draft prospective native inputs and split manifest; never releases a campaign."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')


def proposed_plan():
    return dict(schema='threeway-prospective-inputs-v1',status='DRAFT_NOT_RELEASED',
        primary_methods=['GAT+RAES','DGN_RAW_KL_LOCAL','GA-Net'],
        topology_seeds=dict(train=list(range(9700101,9700131)),
            development=list(range(9700201,9700213)),test=list(range(9700301,9700325))),
        training_seeds=list(range(9701001,9701011)),training_episodes_per_seed=100,
        training_exposure=dict(decisions_per_episode=13,updates_per_transition=1,
            update_batch_size=1,checkpoint_selection='FINAL_FIXED_BUDGET',
            collection='METHOD_OWN_POLICY_FROZEN_WITHIN_EPISODE_UPDATED_AFTER_EPISODE'),
        family_cycle=['spread','clustered','hidden_pair'],
        offered_base_mbps=[8,16,24],packet_replicates=2,
        workload_modes=['constant','piecewise_positive'],
        dynamic_controls=['RAES_STALE_X2','RAES_BATCH_X3_B4','RAES_INCREMENTAL'],
        diagnostic_controls=['FIXED_INITIAL','RANDOM','PRESSURE_ONLY_M0'],
        seed_namespace=dict(geometry=[9700101,9700324],training=[9701001,9701010],
            native_train=[9710000,9719999],native_development=[9720000,9729999],
            native_test=[9730000,9739999],engineering=[9790000,9799999]),
        physical=dict(agents=15,channels=[36,40,44,48],fixed_power=True,phy='802.11a',
            sim_seconds=10,traffic_start_seconds=1,traffic_stop_seconds=9.9,
            report_interval_seconds=.7,payload_bytes=1200,mobility=False,burst=False),
        resources=dict(serial_jobs=True,cpu_only=True,torch_intraop_threads=1,
            torch_interop_threads=1,tf_intraop_threads=1,tf_interop_threads=1,
            affinity='PENDING_QUALIFICATION',hardware='i5-13450HX/16logical/Windows11/WSLUbuntu'),
        release_requirements=['actual training/evaluation consumers',
            'prospective behavior exploration and RAES coefficient rules',
            'new geometry native qualification','runtime/source/binary/affinity freeze',
            'primary metrics/paired uncertainty/failure handling',
            'sample size scope and practical margins'])


def validate(plan):
    if plan.get('schema')!='threeway-prospective-inputs-v1' or plan.get('status')!='DRAFT_NOT_RELEASED':
        raise ValueError('this input exporter cannot release a protocol')
    splits=plan.get('topology_seeds',{})
    if set(splits)!={'train','development','test'}:raise ValueError('three disjoint topology splits required')
    seeds=plan.get('training_seeds',[])
    if len(seeds)<10:raise ValueError('at least ten independent training seeds required')
    all_seeds=[seed for values in splits.values() for seed in values]+seeds
    if any(type(seed) is not int or not 9700000<=seed<9710000 for seed in all_seeds):
        raise ValueError('new prospective seed namespace required; historical/reserved seeds forbidden')
    if len(set(all_seeds))!=len(all_seeds) or any(not values for values in splits.values()):
        raise ValueError('duplicate/overlapping/empty split or training seeds')
    count=plan.get('training_episodes_per_seed')
    if type(count) is not int or count<1:raise ValueError('positive prospective episode budget required')
    if plan['family_cycle']!=['spread','clustered','hidden_pair'] or plan['offered_base_mbps']!=[8,16,24]:
        raise ValueError('qualified native family/rate contract required')
    if plan['physical']!=proposed_plan()['physical']:raise ValueError('accepted native physical domain required')
    if plan['training_exposure']!=proposed_plan()['training_exposure']:
        raise ValueError('unqualified training exposure change')


def scenario(seed,family):
    rng=np.random.Generator(np.random.PCG64(seed))
    scale=25 if family!='clustered' else 15
    positions=np.array([[25+scale*(i%5),25+scale*(i//5),1.5] for i in range(15)],dtype=float)
    positions[:,:2]+=rng.uniform(-6,6,(15,2))
    if family=='hidden_pair':positions[:2]=[[50,50,1.5],[50,54,1.5]]
    loads=rng.uniform(.5,1,15);channels=rng.integers(0,4,15).tolist();tokens=rng.uniform(0,1,15).tolist()
    # These are evaluator/native geometry only. Policies receive measured pilots.
    stas=positions+np.array([2,1,-.5])
    if family=='hidden_pair':stas[:2]=[[20,50,1],[80,54,1]]
    geometry=dict(aps=positions.tolist(),stas=stas.tolist())
    digest=hashlib.sha256(json.dumps(geometry,sort_keys=True,allow_nan=False).encode()).hexdigest()
    multipliers=rng.choice([.5,1,1.5],size=(4,15));multipliers[0]=1
    return positions,loads,channels,tokens,digest,multipliers


def inventory(folder):return {p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*')
    if p.is_file() and p!=folder/'artifact_hashes.json'}


def export(plan,output):
    validate(plan);output=Path(output).resolve()
    if HERE not in output.parents or output.exists():raise ValueError('fresh owned prospective input output required')
    output.mkdir(parents=True);write(output/'plan.json',plan)
    (output/'campaign_inputs.py').write_bytes(Path(__file__).read_bytes())
    scenarios={};seen=set()
    for split,seeds in plan['topology_seeds'].items():
        for index,seed in enumerate(seeds):
            family=plan['family_cycle'][index%3]
            positions,loads,channels,tokens,digest,multipliers=scenario(seed,family)
            if digest in seen:raise ValueError('duplicate actual geometry across splits')
            seen.add(digest);name=split+'/'+str(seed);root=output/name;root.mkdir(parents=True)
            np.savetxt(root/'positions.csv',positions,delimiter=',',fmt='%.17g')
            np.savetxt(root/'loads.csv',loads,delimiter=',',fmt='%.17g')
            np.savetxt(root/'qvalues.csv',np.zeros((15,4)),delimiter=',',fmt='%.17g')
            base=plan['offered_base_mbps'][(index//3)%3]
            with (root/'demand.csv').open('x',encoding='utf-8',newline='') as stream:
                stream.write('timeNs,node,nominalBps\n')
                for row,time in zip(multipliers,[1000000000,3100000000,5200000000,7300000000]):
                    for node,(load,multiplier) in enumerate(zip(loads,row)):
                        stream.write(f'{time},{node},{int(base*1000000*load*multiplier)}\n')
            record=dict(split=split,topology_seed=seed,family=family,
                native_scenario='hidden_node' if family=='hidden_pair' else 'baseline',
                geometry_sha256=digest,initial_channels=channels,tokens=tokens,
                development_preview_base_mbps=base,
                input_sha256={p.name:sha(p) for p in root.iterdir()})
            # The preview schedule is one development input, NOT the full
            # load×mode×packet-replicate×training-seed evaluation job matrix.
            write(root/'scenario.json',record);scenarios[name]=record
    manifest=dict(status='DRAFT_INPUT_SNAPSHOT_NOT_CAMPAIGN_RELEASE',scenarios=scenarios,
        generator_sha256=sha(output/'campaign_inputs.py'),plan_sha256=sha(output/'plan.json'),numpy=np.__version__)
    write(output/'scenarios.json',manifest);write(output/'artifact_hashes.json',inventory(output))
    return manifest


def verify(folder):
    folder=Path(folder).resolve();expected=json.loads((folder/'artifact_hashes.json').read_text())
    if inventory(folder)!=expected:raise ValueError('prospective input inventory changed')
    plan=json.loads((folder/'plan.json').read_text());validate(plan)
    manifest=json.loads((folder/'scenarios.json').read_text())
    if manifest['plan_sha256']!=sha(folder/'plan.json'):raise ValueError('plan identity changed')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=export(proposed_plan(),args.output)
    print(json.dumps(dict(status=result['status'],scenarios=len(result['scenarios']))))
