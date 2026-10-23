"""Declared GA-Net reconstruction; see METHOD_SPEC.md for all adaptations.

Histories are complete causal episode prefixes, never a shuffled hidden state.
This module implements learning mechanisms, not a Wi-Fi observation generator.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def soft_target(reward, next_q, next_logp, done, gamma, alpha):
    return reward + gamma * (~done).to(reward.dtype)[..., None] * (next_q - alpha * next_logp)


@torch.no_grad()
def update_target(online, target, *, tau):
    """Published convention: tau multiplies the OLD target."""
    if not 0 <= tau <= 1:
        raise ValueError('tau must be in [0,1]')
    for p, target_p in zip(online.parameters(), target.parameters(), strict=True):
        target_p.mul_(tau).add_(p, alpha=1 - tau)
    for buf, target_buf in zip(online.buffers(), target.buffers(), strict=True):
        target_buf.copy_(buf)


def actor_surrogate(logp, q, baseline, *, alpha):
    return -(logp * (-alpha * logp + q - baseline).detach()).mean()


def counterfactual(critic, history, actions, probabilities, *, lengths=None):
    """Enumerate ONLY a_i; callers freeze the critic/gates for this operation."""
    b, n, k = probabilities.shape
    if actions.shape != (b, n):
        raise ValueError('probabilities/actions dimensions disagree')
    values = []
    for agent in range(n):
        own_values = []
        for action in range(k):
            changed = actions.clone()
            changed[:, agent] = action
            own_values.append(critic(history, changed, lengths=lengths)[:, agent])
        values.append(torch.stack(own_values, -1))
    q = torch.stack(values, 1)
    return q, (probabilities * q).sum(-1)


def hard_gate(logits, *, stochastic):
    if stochastic:
        return F.gumbel_softmax(logits, tau=1., hard=True, dim=-1)[..., 1]
    return (logits.argmax(-1) == 1).to(logits.dtype)


def neighbor_attention(query, key, value, gates):
    """[B,H,N,D] projections; [B,N,N] directed gates; concat head messages."""
    scores = torch.einsum('bhid,bhjd->bhij', query, key) / key.shape[-1]
    # Normalize against KEPT keys; a huge pruned key must not underflow them.
    kept = gates[:, None].detach() > 0
    shift = scores.detach().masked_fill(~kept, -torch.inf).amax(-1, keepdim=True)
    shift = torch.where(kept.any(-1, keepdim=True), shift, scores.detach().amax(-1, keepdim=True))
    # Kept exponents are <=0 and unchanged. Cap only dormant/pruned gate
    # derivatives to avoid overflow in the straight-through backward path.
    mass = (scores - shift).clamp_max(20).exp() * gates[:, None]
    weights = mass / mass.sum(-1, keepdim=True).clamp_min(1.)
    messages = F.relu(torch.einsum('bhij,bhjd->bhid', weights, value))
    b, h, n, d = messages.shape
    return messages.permute(0, 2, 1, 3).reshape(b, n, h * d), weights


def mlp(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_dim))


class LocalActors(nn.Module):
    def __init__(self, agents, observation_dim, actions):
        super().__init__()
        self.agents, self.observation_dim, self.actions = agents, observation_dim, actions
        self.policies = nn.ModuleList([mlp(observation_dim, actions) for _ in range(agents)])

    def forward(self, observation):
        if (observation.ndim != 3 or observation.shape[1:] != (self.agents, self.observation_dim)
                or not observation.is_floating_point() or not torch.isfinite(observation).all()):
            raise ValueError('finite [batch,agents,observation_dim] local observation required')
        return torch.stack([policy(observation[:, i]).softmax(-1)
                            for i, policy in enumerate(self.policies)], 1)


class HardAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(64, 32, batch_first=True, bidirectional=True)
        self.logits = nn.Linear(64, 2)

    def forward(self, encoded, *, stochastic):
        b, n, d = encoded.shape
        if n == 1:
            return encoded.new_zeros(b, 1, 1)
        neighbors = torch.tensor([[j for j in range(n) if j != i] for i in range(n)], device=encoded.device)
        own = encoded[:, :, None].expand(b, n, n - 1, d)
        remote = encoded[:, neighbors]
        pairs = torch.cat([own, remote], -1).reshape(b * n, n - 1, 2 * d)
        recurrent, _ = self.gru(pairs)
        keep = hard_gate(self.logits(recurrent), stochastic=stochastic).reshape(b, n, n - 1)
        return encoded.new_zeros(b, n, n).scatter(2, neighbors[None].expand(b, -1, -1), keep)


class HistoricalCritic(nn.Module):
    def __init__(self, agents, observation_dim, actions):
        super().__init__()
        self.agents, self.observation_dim, self.actions = agents, observation_dim, actions
        self.observation_encoder = nn.Sequential(nn.Linear(observation_dim, 32), nn.ReLU())
        self.history_gru = nn.GRU(32, 32, batch_first=True)
        self.state_action = nn.Sequential(nn.Linear(32 + actions, 32), nn.ReLU())
        self.hard_attention = HardAttention()
        self.query = nn.Linear(32, 128, bias=False)
        self.key = nn.Linear(32, 128, bias=False)
        self.value = nn.Linear(32, 128, bias=False)
        self.q_heads = nn.ModuleList([mlp(32 + 128, 1) for _ in range(agents)])

    def forward(self, history, actions, *, lengths=None, gates=None, return_evidence=False):
        if (history.ndim != 4 or history.shape[2:] != (self.agents, self.observation_dim)
                or history.shape[1] < 1 or not history.is_floating_point()
                or not torch.isfinite(history).all()):
            raise ValueError('finite [batch,time,agents,observation_dim] history required')
        b, t, n, _ = history.shape
        if (actions.shape != (b, n) or actions.dtype not in (torch.int32, torch.int64)
                or torch.any(actions < 0) or torch.any(actions >= self.actions)):
            raise ValueError('complete integer categorical actions required')
        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.long, device=history.device)
        if (lengths.shape != (b,) or lengths.dtype not in (torch.int32, torch.int64)
                or torch.any(lengths < 1) or torch.any(lengths > t)):
            raise ValueError('valid history lengths required')
        obs = self.observation_encoder(history).permute(0, 2, 1, 3).reshape(b * n, t, 32)
        sequence = nn.utils.rnn.pack_padded_sequence(obs, lengths.repeat_interleave(n).cpu(),
                                                    batch_first=True, enforce_sorted=False)
        _, hidden = self.history_gru(sequence)
        encoded = hidden[-1].reshape(b, n, 32)
        local = self.state_action(torch.cat([encoded, F.one_hot(actions.long(), self.actions).to(encoded.dtype)], -1))
        if gates is None:
            gates = self.hard_attention(encoded, stochastic=self.training)
        elif (gates.shape != (b, n, n) or not torch.isfinite(gates).all()
              or torch.any((gates != 0) & (gates != 1)) or torch.any(gates.diagonal(dim1=-2, dim2=-1) != 0)):
            raise ValueError('binary directed gates without self edges required')
        def heads(projection, features):
            return projection(features).reshape(b, n, 4, 32).permute(0, 2, 1, 3)
        message, weights = neighbor_attention(heads(self.query, encoded), heads(self.key, encoded),
                                             heads(self.value, local), gates)
        combined = torch.cat([local, message], -1)
        q = torch.cat([head(combined[:, i]) for i, head in enumerate(self.q_heads)], -1)
        if return_evidence:
            return q, dict(gates=gates, weights=weights, messages=message)
        return q


class Learner:
    """CPU native SAC/counterfactual updates with restartable optimizer/RNG state."""
    def __init__(self, agents, observation_dim, actions, *, gamma=.9, alpha=.1, tau=.99):
        if any(type(v) is not int or v < 1 for v in [agents, observation_dim, actions]):
            raise ValueError('positive integer network dimensions required')
        if not 0 <= gamma <= 1 or not 0 <= tau <= 1 or not 0 <= alpha < float('inf'):
            raise ValueError('invalid learning coefficients')
        self.config = dict(agents=agents, observation_dim=observation_dim, actions=actions,
                           gamma=gamma, alpha=alpha, tau=tau)
        self.actors = LocalActors(agents, observation_dim, actions)
        self.critic = HistoricalCritic(agents, observation_dim, actions)
        self.target_actors = deepcopy(self.actors).eval().requires_grad_(False)
        self.target_critic = deepcopy(self.critic).eval().requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actors.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.updates = 0

    @torch.no_grad()
    def act(self, observation):
        """Target local actors, deterministic categorical argmax, lowest-index ties."""
        return self.target_actors(observation).argmax(-1)

    def train_batch(self, history, actions, rewards, done, next_history, *, lengths=None, next_lengths=None):
        b, t = history.shape[:2]
        n = self.config['agents']
        if (history.device.type != 'cpu' or rewards.shape != (b, n)
                or not rewards.is_floating_point() or not torch.isfinite(rewards).all()
                or done.shape != (b,) or done.dtype != torch.bool):
            raise ValueError('finite CPU rewards and boolean terminal flags required')
        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.long)
        # Current critic validates history, lengths and actions before mutation.
        self.critic.train()
        predicted = self.critic(history, actions, lengths=lengths)
        obs = history[torch.arange(b), lengths - 1]
        next_q = torch.zeros_like(rewards)
        next_logp = torch.zeros_like(rewards)
        live = ~done
        if live.any():
            if next_history is None or next_history.shape[0] != b:
                raise ValueError('nonterminal replay requires next history')
            if next_lengths is None:
                next_lengths = lengths + 1
            if next_lengths.shape != (b,) or next_lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError('next history lengths required')
            for index in torch.where(live)[0].tolist():
                length = int(lengths[index])
                if (int(next_lengths[index]) != length + 1 or next_history.shape[1] < length + 1
                        or not torch.equal(history[index, :length], next_history[index, :length])):
                    raise ValueError('next history must extend the same causal episode prefix by one observation')
            with torch.no_grad():
                following = next_history[live]
                following_lengths = next_lengths[live]
                next_obs = following[torch.arange(int(live.sum())), following_lengths - 1]
                target_prob = self.target_actors(next_obs)
                target_actions = torch.distributions.Categorical(probs=target_prob).sample()
                next_q[live] = self.target_critic(following, target_actions, lengths=following_lengths)
                next_logp[live] = target_prob.gather(-1, target_actions[..., None]).squeeze(-1).clamp_min(1e-12).log()
        targets = soft_target(rewards, next_q, next_logp, done, self.config['gamma'], self.config['alpha']).detach()
        critic_loss = (predicted - targets).square().sum(-1).mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 10., error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        probabilities = self.actors(obs)
        sampled = torch.distributions.Categorical(probs=probabilities).sample()
        logp = probabilities.gather(-1, sampled[..., None]).squeeze(-1).clamp_min(1e-12).log()
        # Fixed deterministic state gates make all enumerated actions comparable.
        self.critic.eval()
        with torch.no_grad():
            all_q, baseline = counterfactual(self.critic, history, sampled, probabilities.detach(), lengths=lengths)
            q = all_q.gather(-1, sampled[..., None]).squeeze(-1)
        self.critic.train()
        actor_loss = n * actor_surrogate(logp, q, baseline, alpha=self.config['alpha'])
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_norm = nn.utils.clip_grad_norm_(self.actors.parameters(), 10., error_if_nonfinite=True)
        self.actor_optimizer.step()
        update_target(self.actors, self.target_actors, tau=self.config['tau'])
        update_target(self.critic, self.target_critic, tau=self.config['tau'])
        self.updates += 1
        return dict(updates=self.updates, critic_loss=float(critic_loss.detach()),
                    actor_loss=float(actor_loss.detach()), critic_grad_norm=float(critic_norm),
                    actor_grad_norm=float(actor_norm), bootstrapped_transitions=int(live.sum()),
                    targets=targets.clone())

    def save(self, path):
        state = dict(schema='ganet-reconstruction-checkpoint-v1', config=self.config,
                     actors=self.actors.state_dict(), critic=self.critic.state_dict(),
                     target_actors=self.target_actors.state_dict(), target_critic=self.target_critic.state_dict(),
                     actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                     updates=self.updates, torch_cpu_rng=torch.get_rng_state())
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['schema'] != 'ganet-reconstruction-checkpoint-v1':
            raise ValueError('unknown GA-Net checkpoint schema')
        learner = cls(**state['config'])
        for name in ['actors', 'critic', 'target_actors', 'target_critic']:
            getattr(learner, name).load_state_dict(state[name], strict=True)
        learner.actor_optimizer.load_state_dict(state['actor_optimizer'])
        learner.critic_optimizer.load_state_dict(state['critic_optimizer'])
        learner.updates = state['updates']
        torch.set_rng_state(state['torch_cpu_rng'])
        return learner
