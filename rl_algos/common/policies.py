import torch
import torch.nn as nn

from typing import List, Union

try:
    from .utils import build_network
except:
    from utils import build_network


class ActorCriticBetaPolicy(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activations: List[str],
                 hidden_dims: List[str],
                 action_high: torch.Tensor,
                 action_low : torch.Tensor,
                 shared_dims: int = 0,
                 **kwargs) -> None:
        super().__init__()

        self.register_buffer("action_scale", (action_high - action_low) )
        self.register_buffer("action_bias",  action_low)

        dims = [input_dim] + hidden_dims + [2*output_dim]
        acts = activations + [None]

        if shared_dims == 0:
            self.features = nn.Identity()
        else:
            s_dims = dims[0:shared_dims+1]
            s_acts = acts[0:shared_dims]
            self.features = build_network(s_dims, s_acts)

        self.pi = build_network(dims[shared_dims:], acts[shared_dims:])             # Policy
        self.vf = build_network(dims[shared_dims:-1]+[1], acts[shared_dims:])       # Value Function

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.vf(self.features(x))
    

    def forward(self, x: torch.Tensor, a: Union[torch.Tensor, None] = None):
        f = self.features(x)
        params = self.pi(f)

        alpha, beta = torch.chunk(params, 2, dim=-1)

        alpha = torch.nn.functional.softplus(alpha) + 1.0  # Avoid instabilities
        beta = torch.nn.functional.softplus(beta)   + 1.0

        # Beta distribution
        probs = torch.distributions.Beta(alpha, beta)
        
        if a is None:
            a_unscaled = probs.rsample()
        else:
            a_unscaled = (a - self.action_bias) / self.action_scale

        return a_unscaled *  self.action_scale +  self.action_bias, probs.log_prob(a_unscaled).sum(1), probs.entropy().sum(1), self.vf(f)


# Combined Modules with optinal shared feature extraction 
class ActorCriticGaussianPolicy(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activations: List[str], 
                 hidden_dims: List[int],
                 action_high: torch.Tensor,
                 action_low : torch.Tensor,
                 shared_dims: int = 0,
                 log_std_max: float = -1.0,
                 log_std_min: float = -10.0,
                 log_std_init: float = -2.0,
                 **kwargs) -> None:
        super().__init__()

        self._log_std_max = log_std_max
        self._log_std_min = log_std_min
        self._log_std = nn.Parameter(torch.ones(output_dim) * log_std_init, requires_grad=True)

        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias",  (action_high + action_low) / 2.0)

        dims = [input_dim] + hidden_dims + [output_dim]
        acts = activations + ["Tanh"]     # Last Activation

        if shared_dims == 0:
            self.features = nn.Identity()
        else:
            s_dims = dims[0:shared_dims+1]
            s_acts = acts[0:shared_dims]
            self.features = build_network(s_dims, s_acts)

        self.pi = build_network(dims[shared_dims:], acts[shared_dims:])                     # Policy
        self.vf = build_network(dims[shared_dims:-1]+[1], acts[shared_dims:-1]+[None])      # Value Function


    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.vf(self.features(x))
    
    def action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean = self.pi(self.features(x))
        mean = mean * self.action_scale + self.action_bias 

        if deterministic:
            return mean
        
        log_std = self._log_std.clamp(self._log_std_min, self._log_std_max).expand_as(mean)
        std = log_std.exp() * self.action_scale

        return  torch.distributions.Normal(mean, std).sample()


    def forward(self, x: torch.Tensor, a: Union[torch.Tensor, None] = None):
        f = self.features(x)
        mean = self.pi(f)
        mean = mean * self.action_scale + self.action_bias
        log_std = self._log_std.clamp(self._log_std_min, self._log_std_max).expand_as(mean)
        std = log_std.exp() * self.action_scale

        probs = torch.distributions.Normal(mean, std)
        if a is None:
            a = probs.rsample()

        return a, probs.log_prob(a).sum(1), probs.entropy().sum(1), self.vf(f)

class ActorCriticDeterministicPolicy(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activations: List[str], 
                 hidden_dims: List[int],
                 action_high: torch.Tensor,
                 action_low : torch.Tensor,
                 shared_dims: int = 0,
                 **kwargs)-> None:
        super().__init__()

        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias",  (action_high + action_low) / 2.0)

        dims = [input_dim] + hidden_dims + [output_dim]
        acts = activations + ["Tanh"]     # Last Activation

        if shared_dims == 0:
            self.features = nn.Identity()
        else:
            s_dims = dims[0:shared_dims+1]
            s_acts = acts[0:shared_dims]
            self.features = build_network(s_dims, s_acts)

        self.pi = build_network(dims[shared_dims:], acts[shared_dims:])
        self.qf = build_network([dims[shared_dims]+output_dim]+dims[shared_dims+1:-1]+[1], acts[shared_dims:-1]+[None])


    def value(self, x: torch.Tensor, a: torch.Tensor):
        return self.qf(torch.cat([self.features(x), a], 1))

    def action(self, x: torch.Tensor):
        return self.pi(self.features(x)) * self.action_scale + self.action_bias 

    def forward(self, x: torch.Tensor, a: Union[torch.Tensor, None] = None):
        f = self.features(x)
        if a is not None:
            a = self.pi(f) * self.action_scale + self.action_bias 
            q = self.qf(torch.cat([x, a], 1))
        return a, q


# Separate Modules for actor(policy) and critic(Q-value) functions
class ActorDeterministic(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activations: List[str],
                 hidden_dims: List[int],
                 action_high: torch.Tensor,
                 action_low : torch.Tensor,
                 **kwargs) -> None:
        super().__init__()

        self.register_buffer("action_scale", (action_high - action_low) / 2.0 )
        self.register_buffer("action_bias",  (action_high + action_low) / 2.0)

        dims = [input_dim] + hidden_dims + [output_dim]
        acts = activations + ["Tanh"]     # Last Activation

        self.pi = build_network(dims, acts)

    def forward(self, x: torch.Tensor):
        return self.pi(x) * self.action_scale + self.action_bias 


class Critic(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activations: List[str],
                 hidden_dims: List[int],
                 **kwargs):
        super().__init__()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        acts = activations + [None]     # Last Activation

        self.qf = build_network(dims, acts)

    def forward(self, x: torch.Tensor, a: Union[torch.Tensor, None] = None):
        if a is not None:
            return self.qf(torch.cat([x, a], 1))
        
        return self.qf(x)
        
class DualCritic(nn.Module):
    def __init__(self,
                 input_dim: int, 
                 output_dim: int,
                 activations: List[str],
                 hidden_dims: List[int],
                 shared_dims: int = 0,
                 **krargs) -> None:
        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]
        acts = activations + [None]     # Last Activation

        if shared_dims == 0:
            self.features = nn.Identity()
        else:
            s_dims = dims[0:shared_dims+1]
            s_acts = acts[0:shared_dims]
            self.features = build_network(s_dims, s_acts)

        self.adv = build_network(dims[shared_dims:], acts[shared_dims:])        # Advantage
        self.vf = build_network(dims[shared_dims:-1]+[1], acts[shared_dims:])   # Value


    def forward(self, x: torch.Tensor):
        f = self.features(x)
        adv = self.adv(f)
        v = self.vf(f)

        return v + (adv - adv.mean())


# ---- Utilities  ----



if __name__ == "__main__":
    input_dim = 4
    output_dim = 2
    activations = ["ReLU", "ReLU", "ReLU"]
    hidden_dims = [120, 120, 84]
    high = torch.full((4,), 2)
    low = torch.full((4,), 0)

    a = ActorCriticDeterministicPolicy(input_dim, output_dim, activations, hidden_dims, high, low, 2)

    print(a)