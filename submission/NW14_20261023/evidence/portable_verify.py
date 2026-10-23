"""Verify supplied ZIP evidence without the author's absolute workspace paths. Standard library only."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

def percentile(values,q):
    if not values:return None
    v=sorted(values);h=(len(v)-1)*q;lo=math.floor(h);hi=math.ceil(h)
    return v[lo]+(v[hi]-v[lo])*(h-lo)

def readrows(archive,name):
    with archive.open(name) as f:
        yield from csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline=''))

def audit(cell,archive,aparchive=None):
    prefix=cell['relative_folder'];demand=collections.defaultdict(float);telem=collections.Counter()
    for r in readrows(archive,prefix+'/receiver.csv.telemetry.csv'):
        node=int(r['node']);demand[node]+=float(r['nominalDemandBytes']);telem[node]+=int(r['deliveredBytes'])
    assert set(demand)==set(range(15)) and all(b>0 for b in demand.values())
    seen=set();delivered={};counts=collections.Counter();bytes_rx=collections.Counter();delays=[];rawsha=hashlib.sha256()
    byflow=collections.defaultdict(collections.Counter)
    with archive.open(prefix+'/receiver.csv.packet-cohort.csv') as f:
        def lines():
            for raw in f:rawsha.update(raw);yield raw.decode('utf-8-sig')
        for r in csv.DictReader(lines()):
            n,uid,size,sent=int(r['node']),int(r['packetUid']),int(r['bytes']),int(r['sentAtNs'])
            assert 0<=n<15 and uid not in seen and size==1200 and 1_000_000_000<=sent<9_900_000_000
            seen.add(uid);status=r['status'];counts[status]+=1;byflow[n][status]+=1
            assert status in ('delivered','undelivered_mac_terminal','undelivered_censored')
            if status=='delivered':
                received=int(r['receivedAtNs']);delay=int(r['delayNs'])
                assert sent<=received<10_000_000_000 and delay==received-sent
                delays.append(delay/1e6);bytes_rx[n]+=size;delivered[(n,uid)]=received
            else:
                assert not r['receivedAtNs'] and not r['delayNs'] and int(r['censorAtNs'])==10_000_000_000
                assert int(r['censorAgeNs'])==10_000_000_000-sent
                assert bool(int(r['macDropEvents']))==(status=='undelivered_mac_terminal')
    assert rawsha.hexdigest()==cell['raw_sha256'],'raw byte hash mismatch'
    assert sum(counts.values())==len(seen)
    for r in readrows(archive,prefix+'/receiver.csv.packet-cohort.csv.summary.csv'):
        n=int(r['node']);c=byflow[n]
        assert int(r['applicationTxPackets'])==sum(c.values())
        assert int(r['applicationDeliveredPackets'])==c['delivered']
        assert int(r['undeliveredMacTerminalPackets'])==c['undelivered_mac_terminal']
        assert int(r['undeliveredCensoredPackets'])==c['undelivered_censored']
    assert all(bytes_rx[n]==telem[n] for n in range(15))
    sat=100*math.fsum(bytes_rx[n]/demand[n] for n in range(15))/15
    goodput=8*sum(bytes_rx.values())/9e6;p95=percentile(delays,.95)
    mean_b=math.fsum(demand.values())/15;mean_z=sat/100
    covariance=math.fsum((demand[n]-mean_b)*(bytes_rx[n]/demand[n]-mean_z) for n in range(15))/15
    identity_goodput=8*15*(mean_b*mean_z+covariance)/9e6
    assert math.isclose(identity_goodput,goodput,abs_tol=1e-8,rel_tol=1e-11)
    assert math.isclose(sat,float(cell['satisfaction_pct']),abs_tol=1e-8,rel_tol=1e-11)
    assert math.isclose(goodput,float(cell['goodput_mbps']),abs_tol=1e-8,rel_tol=1e-11)
    assert math.isclose(p95,float(cell['delivered_delay_p95_ms']),abs_tol=1e-7,rel_tol=1e-10)
    result=dict(identity={k:cell[k] for k in ('method','condition','load_mbps_per_sta','seed')},status='PASS',
        generated_uids=len(seen),final_status_counts=dict(counts),sat_pct=sat,goodput_mbps=goodput,delivered_only_p95_ms=p95,
        mean_nominal_bytes=mean_b,covariance_nominal_bytes_and_delivered_ratio=covariance,covariance_identity_goodput_mbps=identity_goodput)
    if aparchive is not None:
        first={};aphash=hashlib.sha256()
        with aparchive.open(prefix+'/receiver.csv.ap-reception.csv') as f:
            def aplines():
                for raw in f:aphash.update(raw);yield raw.decode('utf-8-sig')
            for r in csv.DictReader(aplines()):
                if r['event']=='rx_outcome' and r['success']=='1':
                    key=(int(r['node']),int(r['packetUid']));at=int(r['timeNs']);first[key]=min(first.get(key,at),at)
        assert aphash.hexdigest()==cell['ap_sha256'],'AP byte hash mismatch'
        result['ap_exact_delivered_uid_time_matches']=sum(first.get(k)==t for k,t in delivered.items())
        result['ap_missing_or_different_first_time']=len(delivered)-result['ap_exact_delivered_uid_time_matches']
        result['ap_success_uids_outside_delivered_application_set']=len(set(first)-set(delivered))
        # Layer-only UIDs are retained, not forced into application deliveries.
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--packets',type=Path,required=True);parser.add_argument('--ap',type=Path)
    parser.add_argument('--all',action='store_true');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert percentile([2.,4.],.95)==3.9 and percentile([3.],.5)==3 and percentile([], .95) is None
    cells=json.loads(args.metadata.read_text(encoding='utf-8'))
    assert len(cells)==660 and len({c['relative_folder'] for c in cells})==660
    selected=cells if args.all else [next(c for c in cells if c['method']=='actuation_hybrid' and c['condition']=='combined' and c['load_mbps_per_sta']=='24' and c['seed']=='9795002')]
    results=[]
    with zipfile.ZipFile(args.packets) as packetzip:
        apzip=zipfile.ZipFile(args.ap) if args.ap else None
        try:
            for c in selected:results.append(audit(c,packetzip,apzip));print(json.dumps(results[-1]),flush=True)
        finally:
            if apzip:apzip.close()
    args.output.write_text(json.dumps(dict(cells=len(results),results=results,
        limits=['Unsigned stored hashes establish archive consistency, not independent historical authenticity.',
                'Final UID ledger is not a replay of every MAC retransmission attempt.',
                'This program verifies supplied data; it does not rerun or build ns-3.']),indent=2),encoding='utf-8')

if __name__=='__main__':main()
