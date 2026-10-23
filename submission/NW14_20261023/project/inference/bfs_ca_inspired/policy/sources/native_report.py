"""Decode the qualified continuous-demand native report, without simulator geometry."""
import numpy as np
import common_observation


def planning_latency_ns(command):
    keys=set(command)&{'planningLatencyNs','planningCpuNs'}
    if len(keys)!=1:raise ValueError('one explicit current or historical planning latency column required')
    value=command[next(iter(keys))]
    if not isinstance(value,str) or not value.isascii() or not value.isdecimal():raise ValueError('unsigned integer planning latency required')
    return int(value)


def integer(value,name):
    if type(value) is not int or value<0:raise ValueError('nonnegative integer '+name+' required')
    return value


def decode(report,calibration,*,tokens,decision_at_ns):
    fresh=calibration['schema']=='physical-sounding-calibration-v2'
    fields={'epoch','observedAtNs','availableAtNs','nodes'}|({'acquisition'} if fresh else set())
    if not isinstance(report,dict) or set(report)!=fields:
        raise ValueError('versioned native observation fields required')
    epoch=integer(report['epoch'],'epoch')
    observed=integer(report['observedAtNs'],'observation time')
    available=integer(report['availableAtNs'],'availability time')
    decision=integer(decision_at_ns,'decision time')
    if not epoch or not observed<available<=decision:
        raise ValueError('complete received report must precede decision')
    if fresh:
        import fresh_calibration
        fresh_calibration.validate_binding(report,calibration,decision)
    n,_,k=calibration['powers_w'].shape
    nodes=report['nodes']
    if not isinstance(nodes,list) or len(nodes)!=n:raise ValueError('complete native node vector required')
    fields={'node','channel','powersW','nominalMbps','previousTxBytes','previousDeliveredBytes',
            'previousNominalBytes','previousWindowStartNs','previousWindowEndNs'}
    current=[];offered=[];delivered=[];nominal=[];windows=[]
    for index,row in enumerate(nodes):
        if not isinstance(row,dict) or set(row)!=fields or integer(row['node'],'node')!=index:
            raise ValueError('ordered native node identities and declared observable fields required')
        channel=integer(row['channel'],'channel')
        if channel>=k:raise ValueError('invalid native channel')
        try:powers=np.asarray(row['powersW'],dtype=float)
        except (ValueError,TypeError) as exc:raise ValueError('invalid received powers') from exc
        if powers.shape!=(n*k,) or not np.array_equal(powers,calibration['powers_w'][index].reshape(-1)):
            raise ValueError('received power payload does not match bound native calibration')
        tx=integer(row['previousTxBytes'],'previous TX bytes')
        rx=integer(row['previousDeliveredBytes'],'previous delivered bytes')
        start=integer(row['previousWindowStartNs'],'previous window start')
        end=integer(row['previousWindowEndNs'],'previous window end')
        demand=row['previousNominalBytes'];load=row['nominalMbps']
        if (type(demand) not in (int,float) or not np.isfinite(demand) or demand<0 or
            type(load) not in (int,float) or not np.isfinite(load) or load<=0 or not start<=end<=observed):
            raise ValueError('invalid causal packet window or configured nominal demand')
        if start==end:
            if epoch!=1 or tx or rx or demand:raise ValueError('empty initial reward window must be explicit')
        elif demand<=0:raise ValueError('positive continuous-demand window required')
        current.append(channel);offered.append(load);delivered.append(rx);nominal.append(demand);windows.append((start,end))
    if len(set(windows))!=1:raise ValueError('native reward windows must match across nodes')
    state=common_observation.snapshot(calibration,current=current,offered_mbps=offered,tokens=tokens,
                                     epoch=decision/1e9,ack_at=available/1e9)
    start,end=windows[0]
    reward=None if start==end else common_observation.packet_reward(
        delivered_bytes=delivered,nominal_bytes=nominal,window_seconds=(end-start)/1e9)
    result=dict(epoch=epoch,observed_at_ns=observed,available_at_ns=available,
                observation=state,previous_reward=reward,previous_window_ns=windows[0])
    if fresh:result['acquisition']=report['acquisition']
    return result
