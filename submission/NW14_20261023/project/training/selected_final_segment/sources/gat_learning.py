"""Common physical-task GAT learner using the unchanged selected DDQN objective."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from candidate_sources.marl_gnn import GAT_Q_Network
from candidate_sources.transition_loss import selected_ddqn_loss

HERE=Path(__file__).resolve().parent
CONFIG=dict(input_dim=11,hidden_dim=64,output_dim=4,num_heads=4,temperature=1.,dropout=.2,edge_dim=0)
OPTIMIZATION=dict(gamma=.9,learning_rate=.001,gradient_norm_clip=1.,target_online_fraction=.05)


def sources():
    names=['gat_learning.py','candidate_sources/marl_gnn.py','candidate_sources/transition_loss.py']
    return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}


def observation(state,adj):
    state=np.asarray(state);adj=np.asarray(adj)
    if state.shape!=(15,11) or state.dtype.kind not in 'iuf' or not np.isfinite(state).all():
        raise ValueError('finite common15x11 state required')
    if (adj.shape!=(15,15) or adj.dtype.kind not in 'biuf' or not np.isfinite(adj).all()
        or np.any((adj!=0)&(adj!=1)) or np.any(np.diag(adj)!=0)):
        raise ValueError('reported fifteen-node binary graph without self edges required')
    x=torch.tensor(state,dtype=torch.float32);a=torch.tensor(adj,dtype=torch.float32)
    if not torch.isfinite(x).all():raise ValueError('state exceeds float32 range')
    return x,a


class Learner:
    def __init__(self,*,seed):
        if type(seed) is not int or not 0<=seed<2**63:raise ValueError('nonnegative integer training seed required')
        self.seed=seed;torch.manual_seed(seed)
        self.online=GAT_Q_Network(**CONFIG).cpu().float()
        self.target=deepcopy(self.online).eval()
        for p in self.target.parameters():p.requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=OPTIMIZATION['learning_rate'])
        self.updates=0

    def train_transition(self,transition):
        try:
            done=transition['done'];actions=np.asarray(transition['intended_actions']);reward=np.asarray(transition['reward'])
            x,adj=observation(transition['state'],transition['adj'])
        except (KeyError,TypeError) as exc:raise ValueError('complete physical transition required') from exc
        if type(done) is not bool:raise ValueError('explicit boolean terminal required')
        if actions.shape!=(15,) or actions.dtype.kind not in 'iu' or np.any((actions<0)|(actions>=4)):
            raise ValueError('complete legal physical intended-action vector required')
        if reward.shape!=(15,) or reward.dtype.kind not in 'iuf' or not np.isfinite(reward).all():
            raise ValueError('finite fifteen-node raw reward required')
        rewards=torch.tensor(reward,dtype=torch.float32)
        if not torch.isfinite(rewards).all():raise ValueError('reward exceeds float32 range')
        # Ignore all next fields for terminal, including deliberately invalid
        # objects. A terminal step never evaluates a fabricated next graph/Q.
        next_inputs=None if done else observation(transition.get('next_state'),transition.get('next_adj'))
        next_online=next_target=None
        if not done:
            self.online.eval();self.target.eval()
            with torch.no_grad():
                next_online=self.online(*next_inputs);next_target=self.target(*next_inputs)
        self.online.train();q=self.online(x,adj);q.retain_grad()
        action_tensor=torch.tensor(actions,dtype=torch.int64)
        loss,targets=selected_ddqn_loss(q,action_tensor,rewards,terminal=done,
            next_online_q=next_online,next_target_q=next_target,gamma=OPTIMIZATION['gamma'])
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.online.parameters(),1.,error_if_nonfinite=True)
        self.optimizer.step();self.updates+=1
        with torch.no_grad():
            for destination,source in zip(self.target.parameters(),self.online.parameters()):
                destination.mul_(.95).add_(source,alpha=.05)
        if any(not torch.isfinite(p).all() for m in (self.online,self.target) for p in m.parameters()):
            raise ValueError('physical GAT update produced nonfinite parameters')
        return dict(update=self.updates,done=done,actions=actions.tolist(),raw_rewards=rewards.tolist(),
            loss=float(loss.detach()),preclip_gradient_norm=float(norm),targets=targets.tolist(),
            current_q=q.detach().tolist(),selected_q_gradient=q.grad.tolist(),
            next_online_q=None if done else next_online.tolist(),next_target_q=None if done else next_target.tolist())

    def save(self,path,*,provenance):
        path=Path(path).resolve()
        if HERE not in path.parents:raise ValueError('checkpoint inside comparison root required')
        if not isinstance(provenance,dict) or not provenance:raise ValueError('explicit training provenance required')
        metadata=json.loads(json.dumps(dict(schema='physical-gat-training-v1',config=CONFIG,optimization=OPTIMIZATION,
            seed=self.seed,updates=self.updates,sources=sources(),provenance=provenance),allow_nan=False))
        with path.open('xb') as stream:
            torch.save(dict(metadata=metadata,online=self.online.state_dict(),target=self.target.state_dict(),
                optimizer=self.optimizer.state_dict(),torch_cpu_rng=torch.get_rng_state()),stream)

    @classmethod
    def load(cls,path):
        payload=torch.load(path,map_location='cpu',weights_only=True);meta=payload['metadata']
        if meta['schema']!='physical-gat-training-v1' or meta['config']!=CONFIG or meta['optimization']!=OPTIMIZATION or meta['sources']!=sources():
            raise ValueError('GAT checkpoint source/config identity differs')
        if type(meta['updates']) is not int or meta['updates']<0:raise ValueError('invalid GAT optimizer-step count')
        learner=cls(seed=meta['seed']);learner.online.load_state_dict(payload['online'],strict=True)
        learner.target.load_state_dict(payload['target'],strict=True);learner.optimizer.load_state_dict(payload['optimizer'])
        if any(not torch.isfinite(p).all() for m in (learner.online,learner.target) for p in m.parameters()):
            raise ValueError('nonfinite GAT checkpoint parameters')
        steps={int(s['step']) for s in learner.optimizer.state.values()}
        if steps!=({meta['updates']} if meta['updates'] else set()):raise ValueError('checkpoint optimizer/update count differs')
        learner.updates=meta['updates'];torch.set_rng_state(payload['torch_cpu_rng']);return learner
