"""Available physical pilot reports -> shared immutable features and packet reward.

This module validates a report, not its external native acquisition provenance.
The campaign runner must bind source_sha256 to actual closed native evidence.
"""
import hashlib
import json
import numbers
import numpy as np


def frozen(array):
    array=np.asarray(array)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


def finite_time(value):
    if isinstance(value,bool) or not isinstance(value,numbers.Real) or not np.isfinite(value) or value<0:
        raise ValueError('finite nonnegative time required')
    return float(value)


def parse_powers(rows, *, agents, channels, epoch, source_sha256):
    epoch=finite_time(epoch)
    if type(agents) is not int or agents<1 or type(channels) is not int or not 1<=channels<=4:
        raise ValueError('positive agent count and one to four channels required')
    if not isinstance(source_sha256,str) or len(source_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError('source CSV SHA-256 required')
    fields={'receiver','sender','channelIndex','slotStartS','slotEndS','arrivalS','pulseEndS','availableAtS','meanPowerW','valid','reason'}
    powers=np.empty((agents,agents,channels));seen=set();times=[];slots={}
    for row in rows:
        if set(row)!=fields or row['valid']!='true' or row['reason']!='received_psd_pilot':
            raise ValueError('complete valid received-PSD pilot report required; missing is unknown')
        try:
            ids=[int(row[name]) for name in ['receiver','sender','channelIndex']]
            if any(str(i)!=str(row[name]) for i,name in zip(ids,['receiver','sender','channelIndex'])):
                raise ValueError('noninteger identity')
            i,j,c=ids
            start,end,arrival,pulse_end,available=[float(row[name]) for name in ['slotStartS','slotEndS','arrivalS','pulseEndS','availableAtS']]
            power=float(row['meanPowerW'])
        except (ValueError,TypeError) as exc:
            raise ValueError('invalid physical CSV field') from exc
        identity=(i,j,c)
        if not (0<=i<agents and 0<=j<agents and 0<=c<channels) or identity in seen:
            raise ValueError('duplicate or invalid report identity')
        if not np.isfinite([start,end,arrival,pulse_end,available,power]).all() or not (0<=start<=arrival<pulse_end<=end<=available<=epoch) or power<=0:
            raise ValueError('invalid power, acquisition interval or unavailable report')
        if not np.isclose(pulse_end-arrival,.000123,rtol=0,atol=1e-12):
            raise ValueError('wrong pilot signature duration')
        slot=(start,end)
        if (j,c) in slots and slots[j,c]!=slot:
            raise ValueError('receiver pilot schedule disagreement')
        slots[j,c]=slot;seen.add(identity);powers[i,j,c]=power;times.append(available)
    if len(seen)!=agents*agents*channels or len(set(times))!=1:
        raise ValueError('incomplete or asynchronous calibration batch')
    ordered=sorted(slots.values())
    if any(left[1]>right[0] for left,right in zip(ordered,ordered[1:])):
        raise ValueError('overlapping pilot slots')
    return powers,times[0],ordered[0][0]


def calibrate(rows, *, agents, channels, epoch, source_sha256):
    powers,available,oldest=parse_powers(rows,agents=agents,channels=channels,epoch=epoch,source_sha256=source_sha256)
    desired=powers[np.arange(agents),np.arange(agents)]
    ratios=powers/desired[:,None,:]
    if not np.isfinite(ratios).all() or not np.allclose(ratios,ratios[:,:,:1],rtol=1e-6,atol=1e-12):
        raise ValueError('channel-independent coupling contract violated')
    coupling=ratios.mean(-1);np.fill_diagonal(coupling,0)
    canonical=dict(schema='physical-sounding-calibration-v1',source_sha256=source_sha256,
                   available_at=available,powers_w=powers.tolist(),coupling=coupling.tolist())
    digest=hashlib.sha256(json.dumps(canonical,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return dict(schema=canonical['schema'],source_sha256=source_sha256,available_at=available,
                powers_w=frozen(powers),coupling=frozen(coupling),digest=digest)


def snapshot(calibration, *, current, offered_mbps, tokens, epoch, ack_at):
    epoch,ack_at=finite_time(epoch),finite_time(ack_at)
    if ack_at>epoch or calibration['available_at']>epoch:
        raise ValueError('unavailable calibration or future ACK')
    n,_,k=calibration['powers_w'].shape
    current=np.asarray(current);offered=np.asarray(offered_mbps);tokens=np.asarray(tokens)
    if current.shape!=(n,) or current.dtype.kind not in 'iu' or np.any(current<0) or np.any(current>=k):
        raise ValueError('complete acknowledged channel indices required')
    for name,array in [('offered',offered),('tokens',tokens)]:
        if array.shape!=(n,) or array.dtype.kind not in 'iuf' or not np.isfinite(array).all() or np.any(array<0):
            raise ValueError('invalid '+name+' report')
    if np.any(tokens>1):raise ValueError('tokens must be in [0,1]')
    load=offered.astype(float)/54.
    coupling=calibration['coupling'];adj=coupling>0
    ahat=coupling*load[None,:]
    pressure=ahat@np.eye(k)[current]
    state=np.column_stack([np.eye(k)[current],adj.sum(-1)/max(n-1,1),load,pressure,tokens])
    if not np.isfinite(state).all():raise ValueError('nonfinite derived features')
    canonical=dict(calibration=calibration['digest'],current=current.tolist(),offered_mbps=offered.tolist(),
                   tokens=tokens.tolist(),epoch=epoch,ack_at=ack_at)
    digest=hashlib.sha256(json.dumps(canonical,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return dict(current=frozen(current),ahat=frozen(ahat),pressure=frozen(pressure),state=frozen(state),
                adj=frozen(adj),ap_ids=frozen(np.arange(n)),digest=digest)


def packet_reward(*, delivered_bytes, nominal_bytes, window_seconds):
    window=finite_time(window_seconds)
    delivered=np.asarray(delivered_bytes);nominal=np.asarray(nominal_bytes)
    if (window<=0 or delivered.ndim!=1 or nominal.shape!=delivered.shape or delivered.size<1
            or delivered.dtype.kind not in 'iu' or np.any(delivered<0)
            or nominal.dtype.kind not in 'iuf' or not np.isfinite(nominal).all() or np.any(nominal<=0)):
        raise ValueError('nonnegative integer delivered bytes and positive nominal demand/window required')
    satisfaction=delivered/nominal
    return dict(reward=frozen(10*np.clip(satisfaction,0,1)),satisfaction=frozen(satisfaction),
                goodput_mbps=frozen(delivered.astype(float)*8/window/1e6))
