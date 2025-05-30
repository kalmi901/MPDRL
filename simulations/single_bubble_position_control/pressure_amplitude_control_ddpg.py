from settings import *
from rl_algos import DDPG
from envs import Pos1B1D as Environment
from envs import ActionSpaceDict, ObservationSpaceDict
import time


# EXPERIMENT PROPERTIES
PROJECT_NAME = "TestRuns"
TRIAL_NAME   = f"DDPG_PA_CONTROL{int(time.time())}"
SAVE_WANDB   = False
LOG_TRAINING = True
LOG_FREQ     = 1
SEED         = 42
RENDER_ENV   = False
EVAL_EPISODES = 1000

# Vectorizazion specific parameters ()
POLICY_UPDATE_FREQ  = 1
GRADIENT_STEPS      = 256
ROLLOUT_STEPS       = 1 
NUM_ENVS            = 128


# ENVIRONMENT PROPERTIES ----
# ACOUSTIC FIELD (Initial values)
NUMBER_OF_HARMONICS     = 2
ACOUSTIC_FIELD_TYPE     = "SW_N"            # Standing Wave located at x = 0
EXCITATION_FREQUENCIES  = [25.0, 50.0]      # [kHz]
PHASE_SHIFT             = [0.0, 0.0]        # [radians]
PRESSURE_AMPLITUDE      = [0.0, 0.0]        # [bar] - initialized with unexcited case

# STATIC FEATURES 
EQUILIBRIUM_RADIUS      = 60.0              # [micron] - the present implementation suppors fixed value
TIME_STEP_LENGTH        = 10                # number acoustic cycles per action
MAX_STEPS_PER_EPISODE   = 128               # number actions per episode
INITIAL_POSITION        = "random"           
TARGET_POSITION         = "random"
APPLY_TERMINATION       = True              # Stop the environment if distance is less than the tolerace
POSITION_TOLERANCE      = 0.01 

# DYNAMIC FEATURES
ACTION_SPACE = ActionSpaceDict(
    PA = {"IDX": [0, 1], "MIN": [0.0, 0.0], "MAX": [1.0, 1.0], "TYPE": "Box"}
    )   
# The agent selects action space components 0 and 1 between 0.0 and 1.0

OBSERVATION_SPACE = ObservationSpaceDict(
    XT = {"IDX": [0], "MIN": [0.05], "MAX": [0.25], "TYPE": "Box"},
    X  = {"IDX": [0], "MIN": [0.05], "MAX": [0.25], "TYPE": "Box", "STACK": 2}
    )
# The agent assumes the target position for bubble-0 is between 0.05 and 0.25 and
# observe the bubble position values (x) where the interest region os between 0.05 and 0.25,
# due to the partial observability of the full state space 2 position values are encoded into the observation


# HYPERPARAMETERS (DDPG)
TOTAL_TIMESTEPS     = 500_000
LEARNING_RATE       = 2.5e-4
TAU                 = 0.005
GAMMA               = 0.99
BUFFER_SIZE         = 20_000
BATTCH_SIZE         = 256
EXPL_NOISE          = 0.1
EXPL_DECAY          = 0.0
NOISE_CLIP          = 0.25
LEARNING_STARTS     = 1000
MAX_GRAD_NORM       = 0.5

# Neural Networks
NET_ARCHS = {
    "pi": {
        "hidden_dims" : [128, 128],
        "activations" : ["Tanh", "Tanh"] },
    "qf": {
        "hidden_dims" : [128, 128],
        "activations" : ["Tanh", "Tanh"] } }


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
        seed=SEED,
        render_env=RENDER_ENV
    )


    model = DDPG(
        venvs=venvs,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        tau=TAU,
        buffer_size=BUFFER_SIZE,
        batch_size=BATTCH_SIZE,
        exploration_noise=EXPL_NOISE,
        exploration_decay_rate=EXPL_DECAY,
        learning_starts=LEARNING_STARTS,
        policy_frequency=POLICY_UPDATE_FREQ,
        rollout_steps=ROLLOUT_STEPS,
        gradient_steps=GRADIENT_STEPS,
        noise_clip=NOISE_CLIP,
        storage_device="cuda",
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

    model.predict(total_episodes=EVAL_EPISODES, save_dir=STAT_DIR, stat_fname=TRIAL_NAME)
    model.save_model(TRIAL_NAME, MODEL_DIR)

if __name__ == "__main__":
    train()


