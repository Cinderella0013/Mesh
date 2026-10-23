"""Frozen within-episode behavior with captured PCG64 state, outside learners."""
from copy import deepcopy
import hashlib
import json
import numpy as np
from policy_timing import measure


class BehaviorPolicy:
    def __init__(self,base,*,rule,seed,epsilon=0):
        if type(base).__name__ not in ('GATRAES','GANet','DGN'):raise ValueError('qualified frozen base policy required')
        if rule not in ('epsilon_greedy','categorical'):raise ValueError('named behavior rule required')
        if rule=='categorical' and type(base).__name__!='GANet':raise ValueError('categorical behavior requires actual local actor probabilities')
        if type(seed) is not int or not 0<=seed<2**63:raise ValueError('explicit nonnegative behavior RNG seed required')
        if isinstance(epsilon,bool) or not np.isscalar(epsilon) or not np.isfinite(epsilon) or not 0<=epsilon<=1:
            raise ValueError('epsilon must be finite in [0,1]')
        if rule=='categorical' and epsilon!=0:raise ValueError('categorical behavior does not take epsilon')
        self.base=base;self.rule=rule;self.seed=seed;self.epsilon=float(epsilon)
        self.rng=np.random.Generator(np.random.PCG64(seed))
        self.identity=base.identity
        self.setup_elapsed_ns=getattr(base,'setup_elapsed_ns',0)
        self.setup_cpu_ns=getattr(base,'setup_cpu_ns',0)

    def checkpoint(self):
        return dict(rule=self.rule,seed=self.seed,epsilon=self.epsilon,rng=deepcopy(self.rng.bit_generator.state))

    @classmethod
    def restore(cls,base,state):
        if set(state)!={'rule','seed','epsilon','rng'} or state['rng'].get('bit_generator')!='PCG64':
            raise ValueError('complete captured PCG64 behavior state required')
        result=cls(base,rule=state['rule'],seed=state['seed'],epsilon=state['epsilon'])
        result.rng.bit_generator.state=deepcopy(state['rng']);return result

    def decide(self,observation):
        greedy,evidence=self.base.decide(observation)
        def select():
            before=hashlib.sha256(json.dumps(self.checkpoint(),sort_keys=True,allow_nan=False).encode()).hexdigest()
            if self.rule=='epsilon_greedy':
                action=np.array(greedy,copy=True)
                if self.epsilon:
                    mask=self.rng.random(15)<self.epsilon
                    random_actions=self.rng.integers(0,4,15)
                    action[mask]=random_actions[mask]
            else:
                probabilities=np.asarray(evidence['probabilities'],dtype=float)
                if probabilities.shape!=(15,4) or np.any(probabilities<0) or not np.isfinite(probabilities).all():
                    raise ValueError('complete finite categorical probabilities required')
                probabilities=probabilities/probabilities.sum(-1,keepdims=True)
                action=np.array([self.rng.choice(4,p=row) for row in probabilities],dtype=np.int64)
            detail=dict(rule=self.rule,seed=self.seed,epsilon=self.epsilon,rng_before_sha256=before)
            return action,detail
        (actions,detail),cost=measure(select)
        return actions,dict(evidence,base_channels=np.asarray(greedy).tolist(),behavior=detail,
            phase_costs=dict(evidence['phase_costs'],behavior_selection=cost))

    def close(self):
        if hasattr(self.base,'close'):self.base.close()
