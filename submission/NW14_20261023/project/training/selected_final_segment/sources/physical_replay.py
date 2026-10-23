"""Causal physical replay: command intention is the action, readback is state.

This engineering adapter accepts the qualified thirteen-epoch continuous-demand
bridge with CBR or a hash-bound, qualified piecewise demand schedule. Gapped
epochs remain rejected rather than silently interpolated or omitted.
"""
import json
from pathlib import Path
import numpy as np
from audit_bridge import rows,sha,require
import common_observation
import native_report
import demand_model


def collect(folder,*,tokens):
    folder=Path(folder).resolve()
    manifest=json.loads((folder/'artifact_hashes.json').read_text())
    actual={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p!=folder/'artifact_hashes.json'}
    require(actual==manifest,'physical replay source inventory/content mismatch')
    receipt=json.loads((folder/'receipt.json').read_text())
    require(receipt['status']=='TERMINAL' and receipt['exit_code']==0,'physical replay job not successfully terminal')
    if 'tokens' in receipt:require(np.array_equal(tokens,receipt['tokens']),'policy token identity differs')
    observations=[];acks={}
    for line in (folder/'stdout.log').read_text().splitlines():
        if line.startswith('THREEWAY_OBS '):observations.append(json.loads(line.split(' ',1)[1]))
        if line.startswith('THREEWAY_ACK '):
            ack=json.loads(line.split(' ',1)[1]);require(ack['epoch'] not in acks,'duplicate complete ACK');acks[ack['epoch']]=ack
    require([o['epoch'] for o in observations]==list(range(1,14)),'unqualified replay epoch gaps/coverage')
    require(set(acks).issubset(range(1,14)),'unmatched physical replay ACK')
    power=folder/'receiver.csv.sounding.csv'
    calibration=common_observation.calibrate(rows(power),agents=15,channels=4,epoch=1,source_sha256=sha(power))
    decoded=[native_report.decode(o,calibration,tokens=tokens,decision_at_ns=o['availableAtNs']) for o in observations]
    commands=rows(folder/'receiver.csv.policy-command.csv')
    require([int(c['epoch']) for c in commands]==list(range(1,14)),'complete attempted command coverage required')
    window_rows=rows(folder/'receiver.csv.corrected.csv')
    windows={(int(w['node']),int(w['windowStartNs']),int(w['windowEndNs'])):w for w in window_rows}
    require(len(windows)==len(window_rows)==195,'replay strict packet window coverage')
    schedule=None;demand_windows=None
    schedule_meta=receipt.get('frozen_inputs',{}).get('demandScheduleFile')
    if schedule_meta is not None:
        path=folder/'inputs/demandScheduleFile.csv'
        require(sha(path)==schedule_meta['sha256'],'frozen prospective demand input changed')
        schedule=demand_model.read(path,nodes=15,stop_ns=10_000_000_000)
        demand_rows=rows(folder/'receiver.csv.demand-windows.csv')
        demand_windows={(int(w['node']),int(w['windowStartNs']),int(w['windowEndNs'])):float(w['nominalBytes']) for w in demand_rows}
        require(len(demand_windows)==len(demand_rows)==195 and set(demand_windows)==set(windows),'dynamic demand/window coverage differs')
    states=common_observation.frozen(np.stack([d['observation']['state'] for d in decoded]))
    nominal=np.array([node['nominalMbps'] for node in observations[0]['nodes']])
    if schedule is None:
        require(all(np.array_equal(nominal,[n['nominalMbps'] for n in o['nodes']]) for o in observations),
                'changing nominal demand requires a bound qualified schedule')
        # Native DataRate is integer bps, transported as rate/1e6. Recover the
        # integer before integrating: a float Mbps roundtrip changes some exact
        # reward bits at higher offered rates. Preserve native integer-product
        # then double-division arithmetic without weakening exact reward checks.
        nominal_bps=np.rint(nominal*1e6)
        require(np.all((nominal_bps>0)&(nominal_bps<=min(2**53,(2**64-1)//700_000_000))) and
                np.array_equal(nominal_bps/1e6,nominal),
                'constant nominal demand must encode qualified integer bps')
    else:
        for report in observations:
            require(all(row['nominalMbps']==demand_model.rate_at(schedule,n,report['observedAtNs'])/1e6 for n,row in enumerate(report['nodes'])),
                    'reported current load disagrees with prospective native schedule')
    transitions=[]
    for index,(report,d,command) in enumerate(zip(observations,decoded,commands)):
        epoch=index+1;start=1_000_000_000+index*700_000_000;end=min(start+700_000_000,10_000_000_000)
        require(report['observedAtNs']==start,'physical replay observation schedule differs')
        require(int(command['observedAtNs'])==start and int(command['availableAtNs'])==report['availableAtNs'],
                'command not bound to available physical report')
        require(int(command['commandSubmitNs'])==report['availableAtNs']+native_report.planning_latency_ns(command),
                'physical command delay does not match native submit time')
        intended=np.array([int(c) for c in command['targets'].split(';')],dtype=np.int64)
        require(intended.shape==(15,) and np.all((0<=intended)&(intended<4)),'incomplete/illegal intended actions')
        delivered=np.array([int(windows[n,start,end]['appRxBytes']) for n in range(15)],dtype=np.int64)
        demand=np.array([int(rate)*(min(end,9_900_000_000)-start)/8e9 for rate in nominal_bps]) if schedule is None else np.array(
            [demand_model.nominal_bytes(schedule,n,start,end) for n in range(15)])
        if schedule is not None:
            require(all(np.isclose(demand[n],demand_windows[n,start,end],rtol=1e-12,atol=1e-6) for n in range(15)),
                    'independent schedule integral differs from native demand ledger')
        reward=common_observation.packet_reward(delivered_bytes=delivered,nominal_bytes=demand,window_seconds=(end-start)/1e9)
        done=index==12;following=None if done else decoded[index+1]
        if following is not None:
            require(following['previous_window_ns']==(start,end),'retrospective reward window does not follow attempted action')
            for n,node in enumerate(observations[index+1]['nodes']):
                require(node['previousDeliveredBytes']==int(delivered[n]) and
                        node['previousTxBytes']==int(windows[n,start,end]['appTxBytes']) and
                        np.isclose(node['previousNominalBytes'],demand[n],rtol=1e-12,atol=1e-6),
                        'transported reward differs from strict packet/demand evidence')
            require(np.array_equal(following['previous_reward']['reward'],reward['reward']),
                    'native available reward and replay reward differ')
        ack=acks.get(epoch);ack_channels=None;ack_statuses=None
        if ack is not None:
            require(set(ack)=={'epoch','availableAtNs','channels','statuses'},'invalid complete ACK fields')
            require(int(command['commandSubmitNs'])<=ack['availableAtNs']<end,'ACK outside this attempted action interval')
            ack_channels=np.array(ack['channels']);ack_statuses=np.array(ack['statuses'])
            require(ack_channels.shape==ack_statuses.shape==(15,) and ack_channels.dtype.kind in 'iu' and
                ack_statuses.dtype.kind in 'iu' and np.all((0<=ack_channels)&(ack_channels<4)) and
                np.all((0<=ack_statuses)&(ack_statuses<=4)),'complete legal ACK/readback required')
            require(np.all(ack_channels[ack_statuses==0]==intended[ack_statuses==0]),'successful ACK differs from attempted action')
            ack_channels=common_observation.frozen(ack_channels);ack_statuses=common_observation.frozen(ack_statuses)
        after=following['observation']['current'] if following is not None else ack_channels
        if ack_channels is not None and after is not None:
            require(np.array_equal(ack_channels,after),'next complete report disagrees with acknowledged physical state')
        transitions.append(dict(epoch=epoch,state=states[index],intended_actions=common_observation.frozen(intended),
            adj=d['observation']['adj'],ahat=d['observation']['ahat'],ap_ids=d['observation']['ap_ids'],
            current_channels=d['observation']['current'],
            next_adj=None if done else following['observation']['adj'],
            next_ahat=None if done else following['observation']['ahat'],
            reward=reward['reward'],satisfaction=reward['satisfaction'],delivered_bytes=common_observation.frozen(delivered),
            nominal_bytes=common_observation.frozen(demand),reward_window_ns=(start,end),
            reward_available_at_ns=10_000_000_000 if done else following['available_at_ns'],
            next_state=None if done else states[index+1],done=done,actual_channels_after=after,
            ack_channels=ack_channels,ack_statuses=ack_statuses,
            resolution='COMPLETE_ACK' if ack is not None else 'TERMINAL_READBACK_UNKNOWN' if done else 'NEXT_COMPLETE_REPORT_RESYNC'))
    return dict(schema='physical-cbr-intended-action-replay-v1' if schedule is None else 'physical-piecewise-intended-action-replay-v1',folder=str(folder),
        tokens=np.asarray(tokens).tolist(),
        source_manifest_sha256=sha(folder/'artifact_hashes.json'),
        builder_sources={p.name:sha(p) for p in [Path(__file__),Path(native_report.__file__),Path(common_observation.__file__),Path(demand_model.__file__)]},
        states=states,transitions=transitions)


def dgn_payload(transition):
    from policy_adapters import dgn_masks
    current=dict(state=transition['state'],current=transition['current_channels'],adj=transition['adj'],
        ahat=transition['ahat'],ap_ids=transition['ap_ids'])
    done=transition['done'];require(type(done) is bool,'explicit boolean physical terminal required')
    following=None if done else dict(state=transition['next_state'],current=transition['actual_channels_after'],
        adj=transition['next_adj'],ahat=transition['next_ahat'],ap_ids=transition['ap_ids'])
    return dict(state=current['state'].tolist(),masks=dgn_masks(current).tolist(),
        actions=transition['intended_actions'].tolist(),rewards=transition['reward'].tolist(),done=done,
        next_state=None if done else following['state'].tolist(),next_masks=None if done else dgn_masks(following).tolist())


def ganet_batch(corpus,indices):
    import torch
    require(corpus['schema'] in ('physical-cbr-intended-action-replay-v1','physical-piecewise-intended-action-replay-v1'),'unqualified replay schema')
    require(len(indices)>0 and all(type(i) is int and 0<=i<len(corpus['transitions']) for i in indices),'valid replay indices required')
    transitions=[corpus['transitions'][i] for i in indices];lengths=torch.tensor([i+1 for i in indices],dtype=torch.long)
    history=torch.zeros(len(indices),int(lengths.max()),15,11)
    live=any(not t['done'] for t in transitions)
    next_lengths=lengths+torch.tensor([not t['done'] for t in transitions],dtype=torch.long)
    next_history=torch.zeros(len(indices),int(next_lengths.max()),15,11) if live else None
    for row,(index,t) in enumerate(zip(indices,transitions)):
        history[row,:index+1]=torch.tensor(corpus['states'][:index+1],dtype=torch.float32)
        if live:next_history[row,:int(next_lengths[row])]=torch.tensor(corpus['states'][:int(next_lengths[row])],dtype=torch.float32)
    return dict(history=history,actions=torch.tensor(np.stack([t['intended_actions'] for t in transitions]),dtype=torch.long),
        rewards=torch.tensor(np.stack([t['reward'] for t in transitions]),dtype=torch.float32),
        done=torch.tensor([t['done'] for t in transitions],dtype=torch.bool),next_history=next_history,
        lengths=lengths,next_lengths=next_lengths)
