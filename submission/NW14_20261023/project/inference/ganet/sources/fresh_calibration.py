"""Moving-batch binding; aggregate energy stays explicitly unknown."""
import hashlib,json
import csv
from pathlib import Path
import numpy as np
import common_observation as common

def validate_binding(report,calibration,decision_at_ns):
    expected=dict(batchEpoch=calibration['batch_epoch'],
        oldestAcquisitionNs=calibration['oldest_acquisition_ns'],readyAtNs=calibration['available_at_ns'],
        aggregateEnergy=dict(valid=False,valueW=None,reason='mixed_channel_window'))
    acquisition=report.get('acquisition')
    if (not isinstance(acquisition,dict) or acquisition!=expected or
        any(type(acquisition.get(k)) is not int for k in ('batchEpoch','oldestAcquisitionNs','readyAtNs')) or
        acquisition['aggregateEnergy'].get('valid') is not False or
        report['epoch']!=calibration['batch_epoch'] or
        not calibration['available_at_ns']<=report['observedAtNs']<report['availableAtNs']<=decision_at_ns):
        raise ValueError('fresh report batch, timing or unknown-energy binding differs')
    require_fresh(calibration,decision_at_ns)

def load_report(folder,report):
    epoch=report.get('epoch')
    if type(epoch) is not int or not 1<=epoch<=13:raise ValueError('qualified fresh epoch required')
    path=Path(folder)/f'receiver.csv.fresh-{epoch}.csv'
    data=path.read_bytes()
    rows=list(csv.DictReader(data.decode('utf-8').splitlines()))
    result=calibrate(rows,agents=len(report['nodes']),channels=4,epoch=report['availableAtNs']/1e9,
        source_sha256=hashlib.sha256(data).hexdigest(),batch_epoch=epoch)
    validate_binding(report,result,report['availableAtNs'])
    return result

def require_fresh(calibration,decision_at_ns):
    if type(decision_at_ns) is not int or decision_at_ns<0:raise ValueError('integer decision timestamp required')
    if (calibration['schema']!='physical-sounding-calibration-v2' or
        decision_at_ns<calibration['available_at_ns'] or
        decision_at_ns-calibration['oldest_acquisition_ns']>700000000):
        raise ValueError('unavailable or expired acquisition including decision latency')

def calibrate(rows,*,agents,channels,epoch,source_sha256,batch_epoch):
    if type(batch_epoch) is not int or batch_epoch<1:raise ValueError('positive batch epoch required')
    rows=list(rows)
    if any(r.get('batchEpoch')!=str(batch_epoch) for r in rows):raise ValueError('mixed or unidentified batch')
    plain=[{k:v for k,v in r.items() if k!='batchEpoch'} for r in rows]
    powers,available,oldest=common.parse_powers(plain,agents=agents,channels=channels,epoch=epoch,source_sha256=source_sha256)
    ratios=powers/powers[np.arange(agents),np.arange(agents)][:,None,:]
    if not np.isfinite(ratios).all():raise ValueError('nonfinite coupling')
    coupling=ratios.mean(-1);np.fill_diagonal(coupling,0)
    canonical=dict(schema='physical-sounding-calibration-v2',aggregation='arithmetic_mean_per_channel_desired_normalized_ratios',
        batch_epoch=batch_epoch,source_sha256=source_sha256,available_at=available,
        available_at_ns=round(available*1e9),oldest_acquisition_ns=round(oldest*1e9),
        powers_w=powers.tolist(),coupling=coupling.tolist())
    result=dict(canonical,digest=hashlib.sha256(json.dumps(canonical,sort_keys=True,allow_nan=False).encode()).hexdigest())
    result.update(powers_w=common.frozen(powers),coupling=common.frozen(coupling))
    require_fresh(result,round(epoch*1e9))
    return result
