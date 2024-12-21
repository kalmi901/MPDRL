from settings import *
from rl_algos import PPO
from envs import Pos1B1D as Environment
from envs import ActionSpaceDict, ObservationSpaceDict
import time


# EXPERIMENT PROPERTIES
PROJECT_NAME = "TestRuns"
TRIAL_NAME   = f"PPO_PA_CONTROL{int(time.time())}"
SAVE_WANDB   = False
LOG_TRAINING = True
LOG_FREQ     = 1
SEED         = 42

# Vectorizazion specific parameters ()
ROLLOUT_STEPS       = 24 
NUM_ENVS            = 512
NUM_UPDATE_EPOCHS   = 80
MINI_BATCH_SIZE     = 256


# ENVIRONMENT PROPERTIES ----
# ACOUSTIC FIELD (Initial values)
NUMBER_OF_HARMONICS     = 2
ACOUSTIC_FIELD_TYPE     = "SW_N"            # Standing Wave located at x = 0
EXCITATION_FREQUENCIES  = [25.0, 50.0]      # [kHz]
PHASE_SHIFT             = [0.0, 0.0]        # [radians]
PRESSURE_AMPLITUDE      = [0.0, 0.0]        # [bar] - initialized with unexcited case

# STATIC FEATURES 
EQUILIBRIUM_RADIUS      = 60.0              # [micron] - the present implementation suppors fixed value
TIME_STEP_LENGTH        = 5                 # number acoustic cycles per action
MAX_STEPS_PER_EPISODE   = 100               # number actions per episode
INITIAL_POSITION        = "random"           
TARGET_POSITION         = "random"
APPLY_TERMINATION       = True              # Halt the environment if distance is less than the tolerace
POSITION_TOLERANCE      = 0.01 

# DYNAMIC FEATURES
ACTION_SPACE = ActionSpaceDict(
    PA = {"IDX": [0, 1], "MIN": [0.0, 0.0], "MAX": [1.0, 1.0], "TYPE": "Box"}
    )   
# The agent select action space components 0 and 1 between 0.0 and 1.0

OBSERVATION_SPACE = ObservationSpaceDict(
    XT = {"IDX": [0], "MIN": [0.05], "MAX": [0.25], "TYPE": "Box"},
    X  = {"IDX": [0], "MIN": [0.05], "MAX": [0.25], "TYPE": "Box", "STACK": 2}
    )
# The agent assumes the target position for bubble-0 is between 0.05 and 0.25 and
# observe the bubble position values (x) where the interest region os between 0.05 and 0.25,
# due to the partial observability of the full state space 2 position values are encoded into the observation


# HYPERPARAMETERS (DDPG)
TOTAL_TIMESTEPS     = 5_000_000
LEARNING_RATE       = 2.50e-4
GAMMA               = 0.99
GAE_LAMDA           = 0.90
CLIP_COEF           = 0.5
CLIP_VLOSS          = True
ENT_COEF            = 0.05
VF_COEF             = 4.0
MAX_GRAD_NORM       = 0.5
TARGET_KL           = None
NORM_ADV            = True
POLICY              = "Beta"        # "Gaussian" or "Beta"


# Neural Networks
NET_ARCHS = {
    "hidden_dims": [256, 256],
    "activations": ["ReLU", "ReLU"],
    "shared_dims": 0}


def train():
    venvs = Environment(
        num_envs=NUM_ENVS,
        R0=EQUILIBRIUM_RADIUS,
        components=NUMBER_OF_HARMONICS,
        ac_type=ACOUSTIC_FIELD_TYPE,
        freqs=EXCITATION_FREQUENCIES,
        pa=PRESSURE_AMPLITUDE,
        phase_shift=PHASE_SHIFT,
        action_space_dict=ACTION_SPACE,
        observation_space_dict=OBSERVATION_SPACE,
        episode_length=MAX_STEPS_PER_EPISODE,
        time_step_length=TIME_STEP_LENGTH,
        target_position=TARGET_POSITION,
        initial_position=INITIAL_POSITION,
        seed=SEED
    )


    model = PPO(
        venvs=venvs,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMDA,
        mini_batch_size=MINI_BATCH_SIZE,
        clip_coef = CLIP_COEF, 
        clip_vloss = CLIP_VLOSS,
        ent_coef = ENT_COEF,
        vf_coef = VF_COEF,
        max_grad_norm = MAX_GRAD_NORM,
        target_kl = TARGET_KL,
        norm_adv = NORM_ADV,
        rollout_steps=ROLLOUT_STEPS,
        num_update_epochs=NUM_UPDATE_EPOCHS,
        cuda=True,
        torch_deterministic=True,
        seed=SEED,
        net_archs=NET_ARCHS,
        policy='Beta'
    )


    if LOG_TRAINING:
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    log_dir=LOG_DIR,
                    project_name=PROJECT_NAME,
                    trial_name=TRIAL_NAME,
                    use_wandb=SAVE_WANDB,
                    log_frequency=LOG_FREQ)
    else:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)



if __name__ == "__main__":
    train()


