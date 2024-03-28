import os
import torch
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except:
    pass

from typing import Dict, Any

def build_torch_network(input_dim,
                        output_dim, 
                        fc_dims, 
                        fc_acts,
                        last_act=None) -> nn.Sequential:
    full_dims = [input_dim] + fc_dims + [output_dim]
    num_layers = len(full_dims)
    layers = []
    # Input + Hidden layers
    for i in range(num_layers-2):
        layers.append(
            nn.Linear(full_dims[i], full_dims[i+1])
        )
        if fc_acts[i] == "ReLU":
            layers.append(nn.ReLU())
        elif fc_acts[i] == "Tanh":
            layers.append(nn.Tanh())
        elif fc_acts[i] == "LeakyReLU":
            layers.append(nn.LeakyReLU())


    # Output Layer --- 
    layers.append(
        nn.Linear(full_dims[-2], full_dims[-1])
    )

    # Activation Function of Last Layers
    # TODO: Sigmoid?
    if last_act is not None:
        if last_act == "ReLU":
            layers.append(nn.ReLU())
        elif last_act == "Tanh":
            layers.append(nn.Tanh())


    return nn.Sequential(*layers)


def process_final_observation(next_obs: torch.Tensor, infos: Dict):
    real_next_obs = next_obs.clone()
    if 'final_observation' in infos.keys():
        real_next_obs[infos['dones']] = infos['final_observation']

    return real_next_obs


class TFWriter(SummaryWriter):
    def __init__(self, log_dir: str, project_name: str, run_name: str, model: Any):
        super().__init__(os.path.join(log_dir, "runs", run_name))

        if not hasattr(model, "metadata"):
            self.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join(str(f"|{param}|{getattr(model, param)}|") for param in model.metadata["hyperparameters"] if hasattr(model, param)))
                )


class WandbWriter(SummaryWriter):
    def __init__(self, log_dir: str, project_name: str, run_name: str, model: Any):
        super().__init__(os.path.join(log_dir, "runs", run_name))

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