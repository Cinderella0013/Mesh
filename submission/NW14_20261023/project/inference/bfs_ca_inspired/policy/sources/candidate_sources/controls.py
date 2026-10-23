"""Synthetic mechanism controls; B counts contiguous batches in fixed risk order."""
import numbers
import numpy as np


def control(Q,current,ahat,ap_ids,*,method,pref,batches=1,eta=.05,lam=1.,mu=0.,delta=0.,epsilon_q=1e-6):
    if method not in ('M0','M1','X2','X3'):raise ValueError('unknown method')
    for name,value in dict(pref=pref,eta=eta,lam=lam,mu=mu,delta=delta,epsilon_q=epsilon_q).items():
        if isinstance(value,bool) or not isinstance(value,numbers.Real) or not np.isfinite(value) or value<0 or (name in ('pref','epsilon_q') and value==0):
            raise ValueError(f'invalid {name}')
    if method=='M0' and (mu!=0 or delta!=0):raise ValueError('M0 has no switching penalty')
    try:
        q=np.asarray(Q);a=np.asarray(ahat);c=np.array(current,copy=True);ids=np.asarray(ap_ids)
        if q.ndim!=2 or min(q.shape)<1 or q.dtype.kind not in 'iuf' or not np.isfinite(q).all():raise ValueError('invalid Q')
        n,k=q.shape
        if c.shape!=(n,) or c.dtype.kind not in 'iu' or np.any(c<0) or np.any(c>=k):raise ValueError('invalid channels')
        if ids.shape!=(n,) or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=n:raise ValueError('invalid AP IDs')
        if a.shape!=(n,n) or a.dtype.kind not in 'iuf' or not np.isfinite(a).all() or np.any(a<0) or np.any(np.diag(a)!=0):raise ValueError('invalid coupling')
    except (TypeError,ValueError) as exc:raise ValueError(f'invalid control input: {exc}') from exc
    if isinstance(batches,bool) or not isinstance(batches,numbers.Integral) or not 1<=batches<=n:raise ValueError('B must be integer in [1,N]')
    if method!='X3' and batches!=1:raise ValueError('only X3 accepts batch configuration')
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            a=a.astype(float);q=q.astype(float);assignment=c.copy()
            recompute=lambda: a@np.eye(k)[assignment]
            initial=recompute()
            order=sorted(range(n),key=lambda i:(-initial[i,c[i]],int(ids[i])))
            if method in ('M0','M1'):q=np.zeros_like(q)
            else:q=(q-q.min(axis=1,keepdims=True))/np.maximum(np.ptp(q,axis=1,keepdims=True),epsilon_q)
            count=n if method in ('M0','M1') else (batches if method=='X3' else 1)
            for group in np.array_split(order,count):
                pressure=recompute()
                for i in group:
                    score=(-pressure[i]/pref if method in ('M0','M1') else q[i]-lam*np.maximum(0,pressure[i]/pref-eta/pref))
                    score=score-mu*(np.arange(k)!=c[i])
                    chosen=int(c[i]) if score[c[i]]==score.max() else int(score.argmax())
                    if score[chosen]-score[c[i]]<=delta:chosen=int(c[i])
                    assignment[i]=chosen
            return assignment
    except FloatingPointError as exc:raise ValueError('nonfinite intermediate arithmetic') from exc
