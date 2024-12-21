import torch
import torch.nn as nn

from typing import List, Union, Optional, Any, Dict, Tuple

try:
    from .utils import build_network
except:
    from utils  import build_network


class ActorCriticRNNBetaPolicy(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activations: List[str],
                 hidden_dims: List[int],
                 rnn_hidden_size: int,
                 num_rnn_layers: int,
                 action_high: torch.Tensor,
                 action_low: torch.Tensor,
                 shared_dims: int = 0,
                 shared_rnn: bool = True,
                 enable_critic_rnn: bool = True,
                 rnn_type: str = 'LSTM',
                 **kwargs) -> None:
        super().__init__()

        self.register_buffer("action_scale", (action_high - action_low) )
        self.register_buffer("action_bias", action_low)

        dims = [input_dim] + hidden_dims + [2*output_dim]
        acts = activations + [None]

        if shared_dims == 0:
            self.features = nn.Identity()
        else:
            s_dims = dims[0:shared_dims+1]
            s_acts = acts[0:shared_dims]
            self.features = build_network(s_dims, s_acts)

        self.rnn_type = rnn_type.upper()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size

        
        if self.rnn_type in ["LSTM", "GRU"]:
            self.rnn_actor = nn.LSTM(dims[shared_dims], rnn_hidden_size, num_rnn_layers) if self.rnn_type == "LSTM" else nn.GRU(dims[shared_dims], rnn_hidden_size, num_rnn_layers)
            dims[shared_dims] = rnn_hidden_size
            if enable_critic_rnn and not shared_rnn:
                self.rnn_critic = nn.LSTM(dims[shared_dims], rnn_hidden_size, num_rnn_layers) if self.rnn_type == "LSTM" else nn.GRU(dims[shared_dims], rnn_hidden_size, num_rnn_layers)
            else:
                self.rnn_critic = None
        else:
            raise ValueError("Unsupported RNN type. Choise either `LSTM` or `GRU`!")
        
        
        self.pi = build_network(dims[shared_dims:], acts[shared_dims:])             # Policy
        self.vf = build_network(dims[shared_dims:-1]+[1], acts[shared_dims:])       # Value Function
        

    def initialize_rnn_state(self, batch_size: int, device: str):
        if self.rnn_type.upper() == "LSTM":
            return (
            torch.zeros((self.num_rnn_layers, batch_size, self.rnn_hidden_size), dtype=torch.float32, device=device),
            torch.zeros((self.num_rnn_layers, batch_size, self.rnn_hidden_size), dtype=torch.float32, device=device),
            )
        elif self.rnn_type.upper() == "GRU":
            return torch.zeros((self.num_rnn_layers, batch_size, self.rnn_hidden_size), dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")

    @staticmethod
    def clone_rnn_state(rnn_state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) \
                        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(rnn_state, tuple):
            # LSTM
            return (rnn_state[0].clone(), rnn_state[1].clone())
        else: 
            # GRU
            return rnn_state.clone()

    @staticmethod
    def _process_sequence(
        rnn_module: Union[nn.LSTM, nn.GRU], x: torch.Tensor, 
        rnn_state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        done_mask: torch.Tensor):
        """
            TODO
        """
        # batch size = n_seq when doing gradient update !!
        num_seq = rnn_state[0].shape[1] if isinstance(rnn_state, tuple) else rnn_state.shape[1]
        x = x.reshape((-1, num_seq, rnn_module.input_size))
        done_mask = done_mask.to(dtype=torch.float32).reshape((-1, num_seq))


        # Avoid for loop if we can.
        # ty sb3
        if torch.all(done_mask == 0.0):
            rnn_out, rnn_state = rnn_module(x, rnn_state)
            rnn_out = torch.flatten(rnn_out, 0, 1)
            return rnn_out, rnn_state

        # Iterate over time steps
        rnn_out = []
        for xt, d in zip(x, done_mask):
            if isinstance(rnn_state, tuple):
                # LSTM
                rnn_state = (
                    (1.0 - d).view(1, -1, 1) * rnn_state[0],
                    (1.0 - d).view(1, -1, 1) * rnn_state[1],
                )
            else:
                # GRU
                rnn_state = (1.0 - d).view(1, -1, 1) * rnn_state
            
            xt, rnn_state = rnn_module(xt.unsqueeze(0), rnn_state)
            rnn_out += [xt]
        
        rnn_out = torch.flatten(torch.cat(rnn_out), 0, 1)
        return rnn_out, rnn_state

    def value(self, x:torch.Tensor,
              rnn_state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
              done_mask: torch.Tensor):
        
        # TODO Sharer / Critic RNN
        f = self.features(x)
        h, _ = self._process_sequence(self.rnn_actor, f, rnn_state, done_mask)
        return self.vf(h)


    def forward(self, x: torch.Tensor, 
                rnn_state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                done_mask: torch.Tensor,
                a: Optional[torch.Tensor] = None):
        
        f = self.features(x)
        h, rnn_state = self._process_sequence(self.rnn_actor, f, rnn_state, done_mask)
        params = self.pi(h)
        
        alpha, beta = torch.chunk(params, 2, dim=-1)

        alpha = torch.nn.functional.softplus(alpha) + 1.0
        beta  = torch.nn.functional.softplus(beta)  + 1.0

        probs = torch.distributions.Beta(alpha, beta)

        if a is None:
            a_unscaled = probs.rsample()
        else:
            a_unscaled = (a - self.action_bias) / self.action_scale
  
        return (a_unscaled * self.action_scale + self.action_bias,
                probs.log_prob(a_unscaled).sum(1), 
                probs.entropy().sum(1),
                self.vf(h), 
                rnn_state )
        

        
if __name__ == "__main__":
    input_dim = 4
    output_dim = 2
    static_dim = 2
    activations = ["ReLU", "ReLU", "ReLU"]
    hidden_dims = [128, 256, 84]
    num_rnn_layers = 2
    rnn_hidden_size = 96
    high = torch.full((4,), 2)
    low = torch.full((4,), 0)
    shared_dims = 1
    rnn_type = "LSTM"



    a = ActorCriticRNNBetaPolicy(input_dim, output_dim, 
                                 activations, hidden_dims, 
                                 rnn_hidden_size, num_rnn_layers,
                                 high, low,
                                 shared_dims,
                                 rnn_type=rnn_type)

    print(a)