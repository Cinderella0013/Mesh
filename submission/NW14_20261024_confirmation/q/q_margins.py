"""Frozen-state Q counterfactuals; never predict a new native trajectory or SAT."""
import csv
import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'planner/source/raes.py'
SOURCE_SHA='cfea06e17708e74422fc35606d1a2fd1c88c1078d3de04c7d03427bbeae1854a'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
spec=importlib.util.spec_from_file_location('margin_frozen_raes',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
plan=module.plan

def select(score,allowed,old,delta):
    chosen=old if old in allowed and score[old]==max(score[c] for c in allowed) else min(allowed,key=lambda c:(-score[c],c))
    if old in allowed and chosen!=old and score[chosen]-score[old]<=delta:chosen=old
    return int(chosen)

def decision_margins(q,current,ahat,ids,graph,movable,coeff,budget,expected):
    action,trace=plan(q,current,ahat,ids,conflict_graph=graph,movable=movable,capture_trace=True,q_budget=budget,**coeff)
    assert action.tolist()==list(expected),'recorded action disagrees with frozen planner'
    neutral,_=plan(np.zeros_like(q),current,ahat,ids,conflict_graph=graph,movable=movable,q_budget=budget,**coeff)
    low=q.min(axis=1,keepdims=True)
    normalized=budget*((q-low)/np.maximum(q.max(axis=1,keepdims=True)-low,1e-6))
    pressure=ahat@np.eye(q.shape[1])[current]
    rows=[]
    for t in trace:
        i=t['node'];old=int(current[i]);chosen=int(action[i]);allowed=t['allowed']
        penalty=coeff['lam']*np.maximum(0.,pressure[i]/coeff['pref']-coeff['eta']/coeff['pref'])
        switching=coeff['mu']*(np.arange(q.shape[1])!=old)
        score=normalized[i]-penalty-switching
        no_q_score=-penalty-switching
        pre=select(score,allowed,old,coeff['delta'])
        no_q_pre=select(no_q_score,allowed,old,coeff['delta'])
        final=pre if movable[i] else old
        local_neutral=no_q_pre if movable[i] else old
        assert final==chosen,'reservation-prefix reconstruction disagrees'
        margin=None
        if chosen!=old and old in allowed and movable[i]:
            margin=float(penalty[old]-penalty[chosen]-(coeff['delta']+coeff['mu']-(normalized[i,chosen]-normalized[i,old])))
            assert margin>=-1e-12,'a switch violated its source score threshold'
        rows.append(dict(node=int(i),current=old,recorded_selected=chosen,allowed=json.dumps(allowed),current_allowed=old in allowed,
            gate_blocked=not bool(movable[i]),empty_mask=bool(t['no_available_channel']),mask_overridden_by_gate=bool(t['mask_overridden_by_gate']),
            q_pre_gate_selected=pre,zero_q_pre_gate_selected=no_q_pre,pre_gate_q_changes_choice=pre!=no_q_pre,
            local_zero_q_selected=local_neutral,local_q_changes_final_choice=chosen!=local_neutral,
            epoch_zero_q_selected=int(neutral[i]),epoch_q_changes_final_choice=chosen!=int(neutral[i]),
            q_values=json.dumps(q[i].tolist()),normalized_q=json.dumps(normalized[i].tolist()),
            pressure=json.dumps(pressure[i].tolist()),pressure_penalty=json.dumps(penalty.tolist()),
            q_advantage_vs_current=float(normalized[i,chosen]-normalized[i,old]),
            penalty_decrease_vs_current=float(penalty[old]-penalty[chosen]),conditional_switch_margin=margin))
        remaining=t['remaining']
        if chosen!=old:
            pressure[remaining,old]-=ahat[remaining,i];pressure[remaining,chosen]+=ahat[remaining,i]
    return rows

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,default=Path('D:/Claude-workspace/.worktrees/v33-hybrid-execution/research/submission_package_20261015/current_results_20261016/ALL660_PER_SEED.csv'))
    reg=parser.parse_args().registry.resolve()
    with reg.open(encoding='utf-8-sig',newline='') as f:registry=[r for r in csv.DictReader(f) if r['method']=='actuation_hybrid']
    allrows=[];epochs=[];runs=[];input_hashes={str(SOURCE):SOURCE_SHA}
    for run in registry:
        base=Path(run['evidence_folder'])
        folder=(base if base.is_absolute() else reg.parent/base)/'native'
        ds=[json.loads(l) for l in (folder/'decisions.jsonl').read_text().splitlines() if l.strip()]
        obs={o['epoch']:o for l in (folder/'stdout.log').read_text().splitlines() if l.startswith('THREEWAY_OBS ') for o in [json.loads(l.split(' ',1)[1])]}
        assert len(ds)==len(obs)==13
        for name in ('decisions.jsonl','stdout.log'):
            p=folder/name;input_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        previous_current=previous_action=None;runrows=[]
        identity=dict(condition=run['condition'],load_mbps_per_sta=run['load_mbps_per_sta'],seed=run['seed'])
        for d in ds:
            nodes=obs[d['epoch']]['nodes'];e=d['policy_evidence'];n=len(nodes)
            powers=np.array([v['powersW'] for v in nodes]).reshape(n,n,4)
            coupling=(powers/powers[np.arange(n),np.arange(n)][:,None,:]).mean(-1);np.fill_diagonal(coupling,0.)
            loads=np.array([v['nominalMbps'] for v in nodes])/54.;ahat=coupling*loads[None,:]
            recovered=ahat/loads[None,:];graph=np.maximum(recovered,recovered.T)>=.01;np.fill_diagonal(graph,False)
            current=np.array([v['channel'] for v in nodes],dtype=int)
            movable=np.ones(n,bool) if previous_current is None else ~((previous_action!=previous_current)&(current!=previous_action))
            assert not e['q_neutral'] and e['q_budget']==.02
            choices=decision_margins(np.array(e['q']),current,ahat,np.arange(n),graph,movable,e['coefficients'],e['q_budget'],d['channels'])
            annotated=[{**identity,'epoch':d['epoch'],**row} for row in choices]
            allrows.extend(annotated);runrows.extend(annotated)
            epochs.append({**identity,'epoch':d['epoch'],'node_choices':n,'local_q_changed_nodes':sum(r['local_q_changes_final_choice'] for r in choices),
                'epoch_q_changed_nodes':sum(r['epoch_q_changes_final_choice'] for r in choices),'charged_latency_ns':d['charged_latency_ns'],
                'inference_elapsed_ns':e['phase_costs']['neural']['elapsed_ns']})
            previous_current,previous_action=current,np.array(d['channels'])
        runs.append({**identity,'epochs':len(ds),'node_choices':len(runrows),'local_q_changed_nodes':sum(r['local_q_changes_final_choice'] for r in runrows),
            'epoch_q_changed_nodes':sum(r['epoch_q_changes_final_choice'] for r in runrows),'gate_holds':sum(r['gate_blocked'] for r in runrows)})
    assert len(runs)==60 and len(epochs)==780 and len(allrows)==11700
    assert len({(r['condition'],r['load_mbps_per_sta'],r['seed'],r['epoch'],r['node']) for r in allrows})==11700
    summary={}
    for name,group in [('all_six_conditions',allrows),('primary_five_stress',[r for r in allrows if r['condition']!='normal']),('normal_reference',[r for r in allrows if r['condition']=='normal'])]:
        summary[name]={'node_choices':len(group),'local_q_changed_final_nodes':sum(r['local_q_changes_final_choice'] for r in group),
            'fixed_epoch_q_changed_final_nodes':sum(r['epoch_q_changes_final_choice'] for r in group),'pre_gate_q_changed_nodes':sum(r['pre_gate_q_changes_choice'] for r in group),
            'gate_blocked':sum(r['gate_blocked'] for r in group),'empty_mask':sum(r['empty_mask'] for r in group),
            'actual_switched_with_current_allowed':sum(r['conditional_switch_margin'] is not None for r in group)}
    result={'status':'PASS_FROZEN_STATE_READBACK','source_sha256':SOURCE_SHA,'runs':60,'epochs':780,'summary':summary,
        'counterfactual_service_scores_computed':False,'new_native_runs':0,'new_inference_or_training':False,
        'limits':['Local Q effect holds the actual reservation prefix fixed. Epoch counterfactual allows a different prefix within that one frozen observation. Neither feeds changes into later native observations.','Changed decisions are not proof of better SAT, fairness, delay or learning generalization.','The 60 known historical runs include a separate Normal reference; two seeds do not constitute independent generalization.']}
    write_csv(ROOT/'Q_NODE_MARGINS_11700.csv',allrows);write_csv(ROOT/'Q_EPOCH_780.csv',epochs);write_csv(ROOT/'Q_RUNS_60.csv',runs)
    (ROOT/'Q_RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (ROOT/'Q_INPUT_SHA256.json').write_text(json.dumps(input_hashes,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
