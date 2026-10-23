"""Selected-action DDQN objective for one already validated transition."""
import math
import numbers
import torch
import torch.nn.functional as F


def selected_ddqn_loss(current_q,actions,raw_rewards,*,terminal,next_online_q=None,next_target_q=None,gamma=.9):
    """Return scalar loss and detached targets; terminal never accesses next Q."""
    if type(terminal) is not bool:raise ValueError('terminal must be bool')
    if isinstance(gamma,bool) or not isinstance(gamma,numbers.Real) or not math.isfinite(gamma) or not 0<=gamma<=1:
        raise ValueError('gamma must be finite in [0,1]')
    if not isinstance(current_q,torch.Tensor) or current_q.ndim!=2 or min(current_q.shape)<1 or not current_q.is_floating_point() or not torch.isfinite(current_q).all():
        raise ValueError('current_q must be finite floating N by K tensor')
    n,k=current_q.shape
    def same_float(value,shape,name):
        if not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=current_q.dtype or value.device!=current_q.device or not torch.isfinite(value).all():
            raise ValueError(f'{name} must be finite with shape {shape} and same dtype/device as current_q')
    same_float(raw_rewards,(n,),'raw_rewards')
    if not isinstance(actions,torch.Tensor) or actions.shape!=(n,) or actions.dtype not in (torch.int32,torch.int64) or actions.device!=current_q.device or torch.any(actions<0) or torch.any(actions>=k):
        raise ValueError('actions must be integer vector in [0,K) on current_q device')
    with torch.no_grad():
        targets=raw_rewards.detach().clamp(-15.,15.)
        if not terminal:
            same_float(next_online_q,(n,k),'next_online_q')
            same_float(next_target_q,(n,k),'next_target_q')
            chosen=next_online_q.argmax(dim=1,keepdim=True)
            targets=targets+gamma*next_target_q.gather(1,chosen).squeeze(1)
        if not torch.isfinite(targets).all():raise ValueError('target arithmetic became nonfinite')
    selected=current_q.gather(1,actions.to(torch.int64).unsqueeze(1)).squeeze(1)
    loss=F.smooth_l1_loss(selected,targets,beta=1.,reduction='mean')
    if not torch.isfinite(loss):raise ValueError('loss arithmetic became nonfinite')
    return loss,targets
