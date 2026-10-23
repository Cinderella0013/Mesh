"""Offline Hybrid gate audit; stdlib only. Writes only beside this script."""
import csv
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from collections import Counter, defaultdict
from decimal import Decimal

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
OUT = Path(__file__).resolve().parent
REGISTRY = Path(r'D:\Claude-workspace\.worktrees\v33-hybrid-execution\research\submission_package_20261015\current_results_20261016\ALL660_PER_SEED.csv')
ROOT = Path(r'C:\Users\qoqao\.codex\worktrees\advisor-share\Claude-workspace')
COMMIT = 'f5921db0d72edc70e0c6a5163d4f669ed8eb43c5'
SOURCE_REL = 'submission/NW14_20261023/project/source/native_build/'
FILES = ['decisions.jsonl', 'acks.jsonl', 'stdout.log', 'receiver.csv.channel-control.csv',
         'receiver.csv.management.csv', 'receiver.csv.management-state.csv', 'receiver.csv.policy-command.csv']
STATUS = {0:'OK', 1:'STALE', 2:'REJECTED', 3:'BUSY', 4:'ILLEGAL'}
SOURCE_SHA = {'wifi_threeway.cc':'4b31d60c33c8a8b7f59ec5932e5280439ecc4d0246068a627342f12b7df31cd5',
              'management_link.h':'4dc8e3a7f7b4b68955c9cda7531a1a3e77f11128fc84b316416c28c06637fbd6'}
NORMALIZED_SHA = dict(SOURCE_SHA, **{'wifi_threeway.cc':'f2a9931df94d7ef08357c343381d59776057fce2713c886af554f614139bdb9e'})

def sha(data):
    return hashlib.sha256(data).hexdigest()

def csvrows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def unique(rows, key):
    result = {}
    for row in rows:
        k = key(row)
        assert k not in result, ('duplicate identity', k)
        result[k] = row
    return result

def one(rows, label):
    assert len(rows) == 1, (label, len(rows))
    return rows[0]

def time_interval(text):
    # shortcut: channel trace prints six significant digits; use ACK stamp for exact terminal time.
    t = Decimal(text)
    half = Decimal(10) ** (t.adjusted()-5) * Decimal(500000000)
    return int(t*1000000000-half), int(t*1000000000+half)

def writecsv(name, rows):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def selfcheck():
    assert time_interval('1.0795') == (1079495000,1079505000)
    assert unique([{'n':0},{'n':1}], lambda r:r['n'])[1] == {'n':1}
    try:
        unique([{'n':0},{'n':0}], lambda r:r['n'])
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate identity must fail')
    assert (2,3,1) != (2,3,2), 'REPORT and COMMAND namespaces must remain distinct'

def main():
    selfcheck()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,default=REGISTRY)
    parser.add_argument('--source-dir',type=Path,default=ROOT/SOURCE_REL,help='Directory containing pinned wifi_threeway.cc and management_link.h')
    args=parser.parse_args()
    registry=args.registry.resolve()
    registry_bytes = registry.read_bytes()
    selected = [r for r in csvrows(registry) if r['method'] == 'actuation_hybrid']
    assert len(selected) == 60
    unique(selected, lambda r:(r['condition'],r['load_mbps_per_sta'],r['seed']))
    source = {}
    line_map = {'wifi_threeway.cc':[(928,939,'Execute; BUSY and unchanged paths'),(985,990,'scan asserts no pending permanent action'),
                 (1060,1070,'ScanRetune is a separate acquisition action'),(1142,1156,'Begin creates one pending request'),
                 (1159,1181,'Poll quiescence branch sets channels, schedules Finish, then returns'),
                 (1183,1196,'Poll deadline rejection resumes source, clears pending, completes REJECTED'),
                 (1200,1226,'FinishSwitch requires pending and readback; clears pending and completes OK'),
                 (1226,1232,'scan queue namespace offset and completion callback')],
                'management_link.h':[(17,18,'protocol kind and status namespaces'),(105,115,'rx trace precedes synchronous endpoint/coordinator callback'),
                 (151,163,'Submit sends all targets; 250ms coordinator timeout'),(176,185,'ACK matching and ack_complete'),
                 (188,201,'Endpoint rejects STALE before actuator; BUSY guard; synchronous ACK readback'),
                 (203,205,'Timeout clears coordinator state; does not cancel actuator')]}
    for name in line_map:
        p = args.source_dir/name
        b = p.read_bytes()
        assert sha(b.replace(b'\r\n', b'\n'))==NORMALIZED_SHA[name], ('pinned source mismatch',name)
        source[name] = {'path':str(p), 'working_sha256':sha(b), 'commit_blob_sha256':SOURCE_SHA[name],
                        'normalized_sha256':NORMALIZED_SHA[name],
                        'normalized_equal_to_commit':True, 'lines':[{'start':a,'end':z,'meaning':m} for a,z,m in line_map[name]]}
    source['commit'] = COMMIT
    source['git_diff'] = subprocess.check_output(['git','-C',str(ROOT),'diff',COMMIT,'--',*[SOURCE_REL+n for n in line_map]], text=True) if args.source_dir.resolve()==(ROOT/SOURCE_REL).resolve() else ''
    source['comparison']='pinned normalized source SHA256; default source additionally checked with git diff'
    assert not source['git_diff']
    (OUT/'source_hash_line_map.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    requests, held, manifests, runs = [], [], [], []
    totals = Counter()
    for reg in selected:
        run_id = '/'.join(reg[k] for k in ['condition','load_mbps_per_sta','seed'])
        folder=Path(reg['evidence_folder'])
        native = (folder if folder.is_absolute() else registry.parent/folder)/'native'
        assert 'งานเก่าที่ไม่ได้ใช้' not in str(native)
        blobs = {name:(native/name).read_bytes() for name in FILES}
        manifests.extend({'run_id':run_id,'path':str(native/name),'bytes':len(b),'sha256':sha(b)} for name,b in blobs.items())
        data = {name:[json.loads(x) for x in blobs[name].decode('utf-8-sig').splitlines() if x.strip()] for name in FILES[:2]}
        dec = unique(data['decisions.jsonl'],lambda d:d['epoch'])
        ack = unique(data['acks.jsonl'],lambda a:a['epoch'])
        obs = unique([json.loads(x[len('THREEWAY_OBS '):]) for x in blobs['stdout.log'].decode('utf-8-sig').splitlines() if x.startswith('THREEWAY_OBS ')],lambda a:a['epoch'])
        assert set(dec)==set(ack)==set(obs)==set(range(1,14)),run_id
        cc = csvrows(native/FILES[3]); mg=csvrows(native/FILES[4]); states=csvrows(native/FILES[5]); pc=csvrows(native/FILES[6])
        policy=unique(pc,lambda x:int(x['epoch']))
        mgi=defaultdict(list); cci=defaultdict(list); sti=defaultdict(list)
        mgcounts=Counter(x['event'] for x in mg); stcounts=Counter(x['event'] for x in states)
        assert set(mgcounts)=={'enqueue','tx','rx'}
        assert set(stcounts)=={'report_complete','command_submitted','ack_partial','ack_complete'}
        assert [int(x['timeNs']) for x in mg] == sorted(int(x['timeNs']) for x in mg)
        assert [int(x['timeNs']) for x in states] == sorted(int(x['timeNs']) for x in states)
        for x in mg:
            # kind is essential: a REPORT with seq N is never COMMAND/ACK N.
            mgi[(int(x['seq']),int(x['node']),int(x['kind']),x['event'])].append(x)
        for x in cc:
            assert x['schedule']=='external'
            cci[(int(x['opportunity']),int(x['node']))].append(x)
        for x in states: sti[(int(x['seq']),x['event'])].append(x)
        assert set(int(x['seq']) for x in mg)==set(dec)
        assert set(e for e,n in cci)<=set(dec)
        active_intervals=defaultdict(list); local_requests={}
        for epoch in sorted(dec):
            d,a,o,p=dec[epoch],ack[epoch],obs[epoch],policy[epoch]
            choices=unique(d['policy_evidence']['choices'],lambda x:x['node'])
            nodes=unique(o['nodes'],lambda x:x['node'])
            assert set(choices)==set(nodes)==set(range(15))
            assert len(a['statuses'])==len(a['channels'])==len(d['channels'])==15
            assert d['observed_at_ns']==o['observedAtNs']==int(p['observedAtNs'])
            assert d['available_at_ns']==o['availableAtNs']==int(p['availableAtNs'])
            assert d['channels']==list(map(int,p['targets'].split(';')))
            assert sum(x['gate_blocked'] for x in choices.values())==d['policy_evidence']['gate_blocked_count']
            report_complete=int(one(sti[(epoch,'report_complete')],'report_complete')['timeNs'])
            command_submit=int(one(sti[(epoch,'command_submitted')],'command_submitted')['timeNs'])
            ack_complete=int(one(sti[(epoch,'ack_complete')],'ack_complete')['timeNs'])
            assert report_complete==o['availableAtNs'] and command_submit==int(p['commandSubmitNs'])
            assert ack_complete==a['availableAtNs']
            assert len(sti[(epoch,'ack_partial')])==15
            nextobs=obs.get(epoch+1,{}).get('observedAtNs',0)
            assert not nextobs or ack_complete<nextobs
            for n in range(15):
                traces={}
                for kind in [1,2,3]:
                    events=[one(mgi[(epoch,n,kind,event)],(run_id,epoch,n,kind,event)) for event in ['enqueue','tx','rx']]
                    assert int(events[0]['timeNs'])<=int(events[1]['timeNs'])<=int(events[2]['timeNs'])
                    assert len({tuple(x[k] for k in ['packetUid','seq','node','kind','channel','status','stampNs','payloadHash']) for x in events})==1
                    assert all(x['direction']==('down' if kind==2 else 'up') for x in events)
                    traces[kind]=events
                report,command,ar=traces[1],traces[2],traces[3]
                assert int(report[0]['stampNs'])==o['observedAtNs']
                assert int(report[2]['channel'])==nodes[n]['channel']
                assert int(command[0]['timeNs'])==command_submit and int(command[2]['channel'])==d['channels'][n]
                crx=int(command[2]['timeNs']); terminal=int(ar[0]['stampNs']); arx=int(ar[2]['timeNs'])
                assert terminal==int(ar[0]['timeNs']) and crx<=terminal<=arx<=ack_complete
                status=int(ar[2]['status']); assert status==a['statuses'][n] and int(ar[2]['channel'])==a['channels'][n]
                rows=cci[(epoch,n)]; events=[x['status'] for x in rows]
                expected={0:[['unchanged'],['pause_requested','switching_started','changed']],1:[[]],2:[['pause_requested','rejected']]}
                assert events in expected.get(status,[]),(run_id,epoch,n,status,events)
                for row in rows:
                    if row['status'] in ['unchanged','changed','rejected']:
                        lo,hi=time_interval(row['timeS']);assert lo<=terminal<=hi
                        assert (int(row['newChannel'])-36)//4==a['channels'][n]
                if rows and events[0]=='pause_requested':
                    lo,hi=time_interval(rows[0]['timeS']);assert lo<=crx<=hi
                    active_intervals[n].append((crx,terminal,epoch))
                if status==2:
                    assert terminal-crx==200000000
                    assert rows[-1]['oldChannel']==rows[-1]['newChannel'] and rows[-1]['changed']=='0'
                stale_age=crx-int(command[2]['stampNs'])
                prior_seq=max([e for e in range(1,epoch) if local_requests[(e,n)]['status']!='STALE'], default=0)
                if status==1: assert stale_age>200000000 or epoch<=prior_seq
                result={'run_id':run_id,'condition':reg['condition'],'load_mbps_per_sta':reg['load_mbps_per_sta'],'seed':reg['seed'],
                    'request_namespace':'policy_command','request_id':f'{run_id}/policy_command/{epoch}/{n}', 'epoch':epoch,'node':n,
                    'report_packet_uid':report[0]['packetUid'],'command_packet_uid':command[0]['packetUid'],'ack_packet_uid':ar[0]['packetUid'],
                    'report_identity':f'{run_id}/management/REPORT/{epoch}/{n}',
                    'ack_identity':f'{run_id}/management/ACK/{epoch}/{n}',
                    'observed_at_ns':o['observedAtNs'],'available_at_ns':o['availableAtNs'],'command_submit_ns':command_submit,
                    'command_rx_ns':crx,'command_age_at_rx_ns':stale_age,'endpoint_completion_ack_enqueue_ns':terminal,'ack_rx_ns':arx,
                    'ack_complete_ns':ack_complete,'next_observed_at_ns':nextobs,'ack_complete_before_next_observation':bool(not nextobs or ack_complete<nextobs),
                    'observed_channel':nodes[n]['channel'],'target_channel':d['channels'][n],'ack_channel':a['channels'][n],
                    'status':STATUS[status],'gate_blocked':choices[n]['gate_blocked'],'controller_events':';'.join(events),
                    'pause_requested_timeS':next((x['timeS'] for x in rows if x['status']=='pause_requested'),''),
                    'switching_started_timeS':next((x['timeS'] for x in rows if x['status']=='switching_started'),''),
                    'terminal_trace_timeS':rows[-1]['timeS'] if rows else '',
                    'controller_row_numbers':';'.join(str(cc.index(x)+2) for x in rows),'native_folder':str(native)}
                requests.append(result);local_requests[(epoch,n)]=result
                totals['status_'+STATUS[status]]+=1
        for n, intervals in active_intervals.items():
            assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),('overlapping controller pending',run_id,n)
        for (epoch,n),r in local_requests.items():
            if not r['gate_blocked']: continue
            prev=local_requests[(epoch-1,n)]
            outstanding=[e for start,end,e in active_intervals[n] if e<epoch and end>r['observed_at_ns']]
            assert not outstanding
            prevrows=cci[(epoch-1,n)]
            same_req_switches=sum(x['status'] in ['switching_started','changed'] for x in prevrows)
            later_same_req_switches=sum(x['status'] in ['switching_started','changed'] for x in prevrows if time_interval(x['timeS'])[1]>=prev['endpoint_completion_ack_enqueue_ns'])
            intervening_switches=[x for x in cc if int(x['node'])==n and x['status'] in ['switching_started','changed']
                and prev['endpoint_completion_ack_enqueue_ns']<time_interval(x['timeS'])[0] and time_interval(x['timeS'])[1]<r['observed_at_ns']]
            assert r['target_channel']==r['observed_channel']
            assert r['observed_channel']==prev['ack_channel']
            assert prev['status'] in ['REJECTED','STALE']
            assert not same_req_switches and not later_same_req_switches
            assert not intervening_switches
            assert prev['ack_complete_ns']<r['observed_at_ns']
            h=dict(r);h.update({'previous_request_id':prev['request_id'],'previous_epoch':epoch-1,'previous_status':prev['status'],
                'previous_target_channel':prev['target_channel'],'previous_ack_channel':prev['ack_channel'],
                'previous_command_rx_ns':prev['command_rx_ns'],'previous_command_age_at_rx_ns':prev['command_age_at_rx_ns'],
                'previous_terminal_ack_enqueue_ns':prev['endpoint_completion_ack_enqueue_ns'],'previous_ack_rx_ns':prev['ack_rx_ns'],
                'previous_ack_complete_ns':prev['ack_complete_ns'],'previous_controller_events':prev['controller_events'],
                'previous_terminal_trace_timeS':prev['terminal_trace_timeS'],
                'previous_ack_complete_to_observation_ns':r['observed_at_ns']-prev['ack_complete_ns'],
                'older_pending_policy_request_at_observation':len(outstanding),'same_previous_request_switch_events_entire_record':same_req_switches,
                'same_previous_request_later_switch_events':later_same_req_switches,
                'same_node_policy_switch_between_previous_terminal_and_hold_observation':len(intervening_switches),
                'held_request_switch_events_entire_record':sum(x['status'] in ['switching_started','changed'] for x in cci[(epoch,n)]),
                'bounded_conclusion':'REJECTED pre-switch terminal' if prev['status']=='REJECTED' else 'STALE rejected before actuator; earlier pending intervals closed'})
            held.append(h);totals['held_previous_'+prev['status']]+=1
        runs.append({'run_id':run_id,'condition':reg['condition'],'load_mbps_per_sta':reg['load_mbps_per_sta'],'seed':reg['seed'],
                     'node_epochs':195,'held':sum(x['gate_blocked'] for x in local_requests.values()),
                     'management_events':dict(mgcounts),'state_events':dict(stcounts),'normal_reference':reg['condition']=='normal'})
        totals.update({'management_'+k:v for k,v in mgcounts.items()});totals.update({'state_'+k:v for k,v in stcounts.items()})
    assert len(requests)==11700 and len(held)==1301
    assert totals['held_previous_REJECTED']==1251 and totals['held_previous_STALE']==50
    unique(requests,lambda r:r['request_id']);unique(held,lambda r:r['request_id'])
    writecsv('node_epoch_requests.csv',requests);writecsv('held_cases.csv',held);writecsv('input_hashes.csv',manifests)
    aggregate={'method':'actuation_hybrid','registry':str(registry),'registry_sha256':sha(registry_bytes),'run_count':len(selected),
        'node_epochs':len(requests),'held_cases':len(held),'counts':dict(totals),'runs':runs,
        'held_cases_by_condition':dict(Counter(x['condition'] for x in held)),
        'primary_50_runs':{'node_epochs':sum(x['condition']!='normal' for x in requests),'held_cases':sum(x['condition']!='normal' for x in held)},
        'normal_reference_10_runs':{'node_epochs':sum(x['condition']=='normal' for x in requests),'held_cases':sum(x['condition']=='normal' for x in held)},
        'held_current_statuses':dict(Counter(x['status'] for x in held)),
        'held_current_controller_paths':dict(Counter(x['controller_events'] for x in held)),
        'held_previous_ack_complete_to_observation_ns':{'min':min(x['previous_ack_complete_to_observation_ns'] for x in held),'max':max(x['previous_ack_complete_to_observation_ns'] for x in held)},
        'abnormal_events':{k:totals[k] for k in ['state_ack_timeout','state_ack_ignored','management_invalid_rx','management_enqueue_failed']},
        'unmatched_missing_duplicate_reordered_paths':0,
        'normal_reference_runs':[r for r in runs if r['normal_reference']],
        'source_commit':COMMIT,'source_diff':'none', 'selfchecks':'PASS; identity duplicates, kind namespace, six-significant-digit intervals',
        'assertions':'PASS; every input request has unique REPORT/COMMAND/ACK enqueue-tx-rx identity; exact timing and channel cross-check; complete controller paths; no overlapping pending interval; held previous terminal precedes observation; no same-request switching after rejection/hold',
        'limitations':['Recorded paths for these 60 Hybrid runs only; no universal runtime or hardware guarantee.',
            'Channel-control timeS has six significant digits. Exact terminal time is ACK stamp/enqueue under synchronous source callback; trace match uses rounding intervals.',
            'Scan/calibration retunes have separate namespaces and are not permanent policy requests. Channel-control does not emit each scan retune; this audit makes no claim that the radio never retunes after a rejected policy request.',
            'STALE itself does not cancel an older actuator. No older pending policy interval at held observation is established independently from complete controller paths.',
            'Coordinator timeout itself does not cancel actuator work. These records contain no timeout/ignored/incomplete path; unrecorded or alternate timeout paths remain unproven.',
            'No simulator rerun, no scientific code edits, no fixed gate implementation, no new SAT improvement claim; packet metrics reused without recomputation.']}
    (OUT/'aggregate.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
    print(json.dumps({'node_epochs':len(requests),'held_cases':len(held),'counts':dict(totals),'held_by_condition':aggregate['held_cases_by_condition']},indent=2))

if __name__=='__main__':
    main()
