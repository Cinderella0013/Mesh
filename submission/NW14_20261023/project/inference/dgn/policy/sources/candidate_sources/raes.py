"""Pure reservation executor; estimated coupling uses receiver-row/transmitter-column."""
import numpy as np


def plan(Q, current, ahat, ap_ids, *, pref, eta=.05, lam=1., mu=0.,
         delta=0., epsilon_q=1e-6, capture_trace=False):
    """Return a logical channel plan and optional per-reservation audit trace."""
    scalars = dict(pref=pref, eta=eta, lam=lam, mu=mu, delta=delta, epsilon_q=epsilon_q)
    for name, value in scalars.items():
        if np.ndim(value) != 0 or isinstance(value, (bool, str, complex)):
            raise ValueError(f'{name} must be a finite real scalar')
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be a finite real scalar') from exc
        if not np.isfinite(value) or value < 0 or (name in ('pref', 'epsilon_q') and value == 0):
            raise ValueError(f'{name} outside its finite valid domain')
        scalars[name] = value
    pref, eta, lam, mu, delta, epsilon_q = (scalars[x] for x in scalars)
    try:
        if np.iscomplexobj(Q) or np.iscomplexobj(ahat):
            raise ValueError('Q and ahat must be real')
        Q = np.array(Q, dtype=np.float64, copy=True)
        ahat = np.array(ahat, dtype=np.float64, copy=True)
        current = np.array(current, copy=True)
        ap_ids = np.array(ap_ids, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('inputs must be rectangular numeric arrays') from exc
    if Q.ndim != 2 or min(Q.shape) < 1 or not np.isfinite(Q).all():
        raise ValueError('Q must be a nonempty finite N by K matrix')
    n, k = Q.shape
    for name, vector in (('current', current), ('ap_ids', ap_ids)):
        if vector.shape != (n,) or vector.dtype.kind not in 'iu':
            raise ValueError(f'{name} must be an integer vector of length N')
    if np.any(current < 0) or np.any(current >= k):
        raise ValueError('current channel outside [0, K)')
    if len(np.unique(ap_ids)) != n:
        raise ValueError('ap_ids must be distinct')
    if (ahat.shape != (n, n) or not np.isfinite(ahat).all()
            or np.any(ahat < 0) or np.any(np.diag(ahat) != 0)):
        raise ValueError('ahat must be finite nonnegative N by N with zero diagonal')
    assignment = current.copy()
    history = []
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            low = Q.min(axis=1, keepdims=True)
            q = (Q - low) / np.maximum(Q.max(axis=1, keepdims=True) - low, epsilon_q)
            pressure = ahat @ np.eye(k)[current]
            order = sorted(range(n), key=lambda i: (-pressure[i, current[i]], int(ap_ids[i])))
            channels = np.arange(k)
            for m, i in enumerate(order):
                old = int(current[i])
                score = q[i] - lam * np.maximum(0., pressure[i] / pref - eta / pref) - mu * (channels != old)
                chosen = old if score[old] == score.max() else int(score.argmax())
                if chosen != old and score[chosen] - score[old] <= delta:
                    chosen = old
                remaining = order[m + 1:]
                assignment[i] = chosen
                if chosen != old:
                    pressure[remaining, old] -= ahat[remaining, i]
                    pressure[remaining, chosen] += ahat[remaining, i]
                if capture_trace:
                    history.append(dict(node=i, remaining=remaining, assignment=assignment.copy(),
                                        pressure=pressure[remaining].copy()))
    except FloatingPointError as exc:
        raise ValueError('nonfinite intermediate arithmetic; rescale inputs') from exc
    return assignment, history
