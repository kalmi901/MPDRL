from settings import *
from rl_algos import PPO
from envs import Pos2B1D as Environment
from envs import ActionSpaceDict, ObservationSpaceDict
import time
from math import pi

# EXPERIMENT PROPERTIES
PROJECT_NAME = "TestRuns"
TRIAL_NAME   = f"TrialRun_{int(time.time())}"
SAVE_WANDB   = False
LOG_TRAINING = True
LOG_FREQ     = 1
SEED         = 42
RENDER_ON_TRAIN = False


# Vectorizazion specific parameters ()
ROLLOUT_STEPS       = 512 
NUM_ENVS            = 128
NUM_UPDATE_EPOCHS   = 8
MINI_BATCH_SIZE     = 128


# ENVIRONMENT PROPERTIES ----
# ACOUSTIC FIELD (Initial values)
NUMBER_OF_HARMONICS     = 2
ACOUSTIC_FIELD_TYPE     = "SW_A"            # Standing Wave Antinode located at x = 0
EXCITATION_FREQUENCIES  = [25.0, 250.0]     # [kHz]
PHASE_SHIFT             = [0.0, 0.0]        # [radians]
PRESSURE_AMPLITUDE      = [0.0, 0.0]        # [bar] - initialized with unexcited case

# STATIC FEATURES 
EQUILIBRIUM_RADIUS      = [50.0, 50.0]      # [micron] - the present implementation suppors fixed value
TIME_STEP_LENGTH        = 50                # number acoustic cycles per action
MAX_STEPS_PER_EPISODE   = 512               # number actions per episode
INITIAL_POSITION        = "random"           
TARGET_POSITION         = 0.0
APPLY_TERMINATION       = True              # Halt the environment if distance is less than the tolerace
POSITION_TOLERANCE      = 0.01
DISTANCE_LIMIT          = [0.2, 0.25] 
REWARD_WEIGHTS          = [0.5, 0.5, 0.00]   # Target position, distance penalty, intensity penalty
REWARD_EXPS             = 0.8


ACTION_SPACE = ActionSpaceDict(
    PA = {"IDX": [0, 1], "MIN": [0.0, 0.0], "MAX": [1.0, 1.0], "TYPE": "Box"},
    PS = {"IDX": [0, 1], "MIN": [-0.5*pi, -0.5*pi], "MAX": [0.5*pi, 0.5*pi], "TYPE": "Box"}
    ) 

OBSERVATION_SPACE = ObservationSpaceDict(
    XT = {"IDX": [0], "MIN": [-0.15], "MAX": [0.15], "TYPE": "Box"},
    X  = {"IDX": [0, 1], "MIN": [-0.50, -0.50], "MAX": [0.50, 0.50], "TYPE": "Box", "STACK": 2}
    )


# HYPERPARAMETERS (DDPG)
TOTAL_TIMESTEPS     = 5_000_000
LEARNING_RATE       = 2.50e-4
GAMMA               = 0.99
GAE_LAMDA           = 0.95
CLIP_COEF           = 0.2
CLIP_VLOSS          = True
ENT_COEF            = 0.05
VF_COEF             = 1.0
MAX_GRAD_NORM       = 2.5
TARGET_KL           = None
NORM_ADV            = False


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
        min_distance=DISTANCE_LIMIT[0],
        max_distance=DISTANCE_LIMIT[1],
        reward_weight=REWARD_WEIGHTS,
        reward_exp=REWARD_EXPS,
        render_env=RENDER_ON_TRAIN,
        apply_termination=APPLY_TERMINATION,
        seed=SEED,
        positive_terminal_reward=MAX_STEPS_PER_EPISODE,
        negative_terminal_reward=-MAX_STEPS_PER_EPISODE
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
        net_archs=NET_ARCHS
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
    try:
        train()
    except KeyboardInterrupt:
        print("Exit")
