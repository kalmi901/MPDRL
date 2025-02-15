import os
import torch
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except:
    pass

from typing import Dict, Any, List

def process_final_observation(next_obs: torch.Tensor, infos: Dict):
    real_next_obs = next_obs.clone()
    if 'final_observation' in infos.keys():
        real_next_obs[infos['dones']] = infos['final_observation']

    return real_next_obs

def save_model(model: nn.Module, path: str, fname: str):
    torch.save(model.state_dict(), os.path.join(path, fname))

def load_model(model: nn.Module, path:str, fname: str):
    model.load_state_dict(torch.load(os.path.join(path, fname)))

ACTIVATIONS = ["ReLU", "Tanh", "lReLU"]
def get_activation(activation: str) ->torch.nn.Module:
    if activation == ACTIVATIONS[0]:
        return nn.ReLU()
    elif activation == ACTIVATIONS[1]:
        return nn.Tanh()
    elif activation == ACTIVATIONS[2]:
        return nn.LeakyReLU()
    else:
        print(f"Err: Invalid activation function name: {activation}!")
        exit()


def build_network(dims: List[int], activations: List[str]) -> torch.nn.Module:
    layers = []
    for i, act in enumerate(activations):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if act is not None:
            layers.append(get_activation(act))

    return nn.Sequential(*layers)

class TFWriter(SummaryWriter):
    def __init__(self, log_dir: str, project_name: str, run_name: str, model: Any):
        super().__init__(os.path.join(log_dir, run_name))

        if hasattr(model, "metadata"):
            self.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join(str(f"|{param}|{getattr(model, param)}|") for param in model.metadata["hyperparameters"] if hasattr(model, param)))
                )


class WandbWriter(SummaryWriter):
    def __init__(self, log_dir: str, project_name: str, run_name: str, model: Any):
        super().__init__(os.path.join(log_dir, run_name))

        try:
            entity = os.environ["WANDB_USER"]
        except KeyError:
            raise KeyError ("WANDB User does not found. Please add WANDB_USER=username to your environment variables ")
        
        wandb.init(
            project=project_name,
            name=run_name,
            entity=entity,
            config={f"{param}": getattr(model, param) for param in model.metadata["hyperparameters"] if hasattr(model, param) } if hasattr(model, "metadata") else {},
            dir=log_dir)
        

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False):
        super().add_scalar(tag, scalar_value, global_step, walltime, new_style, double_precision)

        wandb.log({tag: scalar_value}, step=global_step)
    

    def close(self):
        super().close()
        wandb.finish()