"""Numerical train-only scale rule; native measurement provenance is audited separately."""
import numpy as np


def coefficients(measurements,*,training_topologies):
    expected=list(training_topologies)
    if not expected or len(set(expected))!=len(expected):raise ValueError('distinct nonempty training topologies required')
    seen=set();values=[]
    for record in measurements:
        seed=record.get('topology_seed')
        if record.get('split')!='train' or seed not in expected or seed in seen:
            raise ValueError('one training-only measurement per declared geometry required')
        pressure=np.asarray(record.get('pressure'))
        if (pressure.size==0 or pressure.dtype.kind not in 'iuf' or not np.isfinite(pressure).all()
            or np.any(pressure<0)):
            raise ValueError('finite nonnegative measured pressure samples required')
        values.extend(pressure[pressure>0].tolist());seen.add(seed)
    if seen!=set(expected) or not values:raise ValueError('complete positive training calibration coverage required')
    pref=float(np.quantile(np.asarray(values,dtype=np.float64),.95,method='linear'))
    if not np.isfinite(pref) or pref<=0:raise ValueError('positive finite reference pressure required')
    return dict(pref=pref,eta=.05*pref,lam=1.,mu=.02,delta=.01)
