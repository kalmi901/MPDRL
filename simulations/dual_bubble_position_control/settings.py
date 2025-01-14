import sys
import os
sys.path.append("..\MPDRL")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
WORK_DIR = os.path.join(ROOT_DIR, "dual_bubble_position_control")

LOG_DIR   = os.path.join(WORK_DIR, "runs")
MODEL_DIR = os.path.join(WORK_DIR, "models")
STAT_DIR  = os.path.join(WORK_DIR, "statistics")