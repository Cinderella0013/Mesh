"""Check the complete six-method grid and optionally recompute supplied packet evidence."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import zipfile
from portable_verify import audit

ROOT = Path(__file__).resolve().parent
METHODS = ('gat','actuation_hybrid','actuation_qneutral','ganet','dgn','bfs_ca_inspired')
CONDITIONS = ('normal','mobility','hidden_node','burst','adjacent_channel','combined')
KEYS = ('method','condition','load_mbps_per_sta','seed')

def check_tables():
    metadata = json.loads((ROOT/'PORTABLE_CELLS_360.json').read_text(encoding='utf-8'))
    with (ROOT/'tables/RESULTS_360.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    expected = set(itertools.product(METHODS,CONDITIONS,('8','12','16','20','24'),('9795002','9795003')))
    assert len(rows)==len(metadata)==360
    assert {tuple(r[k] for k in KEYS) for r in rows} == expected
    assert {tuple(r[k] for k in KEYS) for r in metadata} == expected
    lookup={tuple(r[k] for k in KEYS):r for r in rows}
    for c in metadata:
        r=lookup[tuple(c[k] for k in KEYS)]
        assert r['status']=='FULL_INTEGRITY_PASS'
        for k in ('satisfaction_pct','goodput_mbps','delivered_delay_p95_ms'):
            assert math.isclose(float(r[k]),float(c[k]),abs_tol=1e-9)
    with (ROOT/'tables/RAW_PER_FLOW_5400.csv').open(encoding='utf-8-sig',newline='') as f: flows=list(csv.DictReader(f))
    assert len(flows)==5400
    assert {tuple(r[k] for k in KEYS)+(r['node'],) for r in flows} == {k+(str(n),) for k in expected for n in range(15)}
    for identity in expected:
        g=[r for r in flows if tuple(r[k] for k in KEYS)==identity]
        sat=100*math.fsum(int(r['bytesDelivered'])/float(r['nominal_bytes']) for r in g)/15
        mbps=8*sum(int(r['bytesDelivered']) for r in g)/9e6
        r=lookup[identity]
        assert math.isclose(sat,float(r['satisfaction_pct']),abs_tol=1e-8)
        assert math.isclose(mbps,float(r['goodput_mbps']),abs_tol=1e-8)
    return metadata

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packets',type=Path);p.add_argument('--ap',type=Path)
    p.add_argument('--all',action='store_true',help='Recompute every cell; otherwise one disclosed sample.')
    p.add_argument('--output',type=Path,default=ROOT/'VERIFICATION_LOCAL.json')
    args=p.parse_args();cells=check_tables();results=[]
    if args.packets:
        selected=cells if args.all else [c for c in cells if (c['method'],c['condition'],c['load_mbps_per_sta'],c['seed'])==('actuation_hybrid','combined','24','9795002')]
        with zipfile.ZipFile(args.packets) as packets:
            aps=zipfile.ZipFile(args.ap) if args.ap else None
            try:
                for c in selected:
                    result=audit(c,packets,aps)
                    if aps: assert result['ap_missing_or_different_first_time']==0
                    results.append(result)
            finally:
                if aps: aps.close()
    report={'table_grid_cells':360,'per_flow_records':5400,'table_metrics_recomputed':360,'raw_packet_cells_recomputed':len(results),'results':results,'native_runs_started':0,'limits':['Table arithmetic and supplied packet evidence are checked; this does not independently rebuild, train or rerun ns-3.','Two known seeds and post-result presentation selection do not prove statistical superiority or fresh generalization.','Delivered-only delay must be read with terminal and censored packet counts.']}
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}))

if __name__=='__main__': main()
