import sys
import os
sys.path.append("..\MPDRL")

import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import Any, List, Dict, DefaultDict

plt.rcParams.update({'axes.facecolor': '#F0F0F0', 'figure.facecolor': '#F0F0F0'})

MAX_COMPONENTS = 5

@dataclass
class control_configureation:
    num_bubbles: int = 1
    num_harmonic_components: int = 2
    dim: int = 1
    ac_type: str = "SW_N"
    freq = [25.0 * 1e3, 50.0 * 1e3]
    pa = [0.0, 0.0]
    ps = [0.0, 0.0] 
    pa_idx = [0, 1]
    ps_idx = [2, 3]
    pa_len = 0
    ps_len = 0
    xlim = []
    x_res = 50
    tspan = 50
    t_res = 50
    setup = False




def setup_acoustic_field(num_components: int, freq: List[int], ac_field:str):
    global _PA

# Static Constants
    w = np.array([2.0 * np.pi * f for f in freq])     # Angular Frequency
    k = np.array([f / 1500 for f in w ])              # Wave Number
    lr = 1500 / freq[0]                     # Wave length
    w_ref = 1.0 / w[0]
    lr_ref = lr / (2 * np.pi )
    #self.sp[2] = lr / (2 * np.pi) 
    #self.dp[3*self._k + i] = 2.0 * np.pi * self._FREQ[i] / CL  

    if ac_field == "CONST":
        def _PA(pa, ps, x, t):
            t = np.asarray(t, dtype=np.float32)  # Ensure t is an array
            x = np.asarray(x, dtype=np.float32)  # Ensure x is an array

            tt, xx = np.meshgrid(t, x, indexing="ij")
            pp = np.zeros_like(tt, dtype=np.float32)

            for i in range(num_components):
                pp += pa[i] * np.sin(2 * np.pi * w_ref * w[i] * tt + ps[i])
            return tt, xx, pp
        
        return _PA
    
    elif ac_field == "SW_A":
        def _PA(pa, ps, x, t):

            t = np.asarray(t, dtype=np.float32)  # Ensure t is an array
            x = np.asarray(x, dtype=np.float32)  # Ensure x is an array

            tt, xx = np.meshgrid(t, x, indexing="ij")
            pp = np.zeros_like(tt, dtype=np.float32)

            for i in range(num_components):
                pp += pa[i]  * np.cos(2*np.pi*lr_ref*k[i] * xx + ps[i]) \
                             * np.sin(2*np.pi*w_ref*w[i] * tt + ps[i])
            return tt, xx, pp
        
        return _PA

    elif ac_field == "SW_N":
        def _PA(pa, ps, x, t):

            t = np.asarray(t, dtype=np.float32)  # Ensure t is an array
            x = np.asarray(x, dtype=np.float32)  # Ensure x is an array

            tt, xx = np.meshgrid(t, x, indexing="ij")
            pp = np.zeros_like(tt, dtype=np.float32)

            for i in range(num_components):
                pp += pa[i]  * np.sin(2*np.pi*lr_ref*k[i] * xx + ps[i]) \
                             * np.sin(2*np.pi*w_ref*w[i] * tt + ps[i])
            return tt, xx, pp
        
        return _PA


def draw_figure(canvas, figure):
    """Draw a matplotlib figure on a Tkinter canvas."""
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def load_trajectories(file_path):
    """Load trajectories from a picke file."""
    trajectories = []
    with open(file_path, "rb") as f:
        while True:
            try:
                trajectory = pickle.load(f)
                trajectories.append(trajectory)
            except EOFError:
                break
    return trajectories

def create_plot(trajectory, config, acoustic_field):
    """Create a matplotlib plot for the given trajectory with 3 subplots."""
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 6))  # Három subplot egy oszlopban

    # Plot 1: Bubble Radius
    time = np.hstack(trajectory.dense_time)
    for i in range(config.num_bubbles):
        radius = np.hstack(trajectory.dense_states[i])
        axs[0].plot(time, radius, label=f"$R_{i}$")
    axs[0].set_title("Bubble Radius Over Time")
    axs[0].set_xlabel(r"Time ($\tau$))")
    axs[0].set_ylabel(r"Radius")
    axs[0].legend()

    # Plot 2: Bubble Position
    for i in range(config.num_bubbles):
        position = np.hstack(trajectory.dense_states[i+config.num_bubbles])
        axs[1].plot(time, position, label=f"Position $x_{i}$")
    axs[1].set_title("Bubble Position Over Time")
    axs[1].set_xlabel(r"Time ($\tau$)")
    axs[1].set_ylabel(r"Position ($x/\lambda_0)$")
    axs[1].set_ylim(config.xlim[0], config.xlim[1])
    axs[1].legend()

    # Plot 3: Acoustic Pressure
    for i in range(trajectory.episode_length):
        t = np.linspace(i * config.tspan, (i+1) * config.tspan, config.t_res) 
        x = np.linspace(*config.xlim, config.x_res)
        for idx in range(0, config.pa_len):
            config.pa[config.pa_idx[idx]] = trajectory.actions[i][idx]
        for idx in range(config.ps_len):
            config.ps[config.ps_idx[idx]] = trajectory.actions[i][idx + config.pa_len] * np.pi

        tt, xx, pp = acoustic_field(config.pa, config.ps, x, t)
       # Create a contour plot for the acoustic field
        c = axs[2].contourf(tt, xx, pp, levels=50, cmap="plasma", linestyles="None")
    axs[2].set_title("Acoustic Field")
    axs[2].set_ylabel("Position (x)")
    axs[2].set_xlabel("Time (t)")
    axs[2].grid(False)

    # Add a colorbar outside the loop to avoid redundancy
    #fig.colorbar(c, ax=axs[2], label="Pressure Amplitude")

    # Stílus és elrendezés
    for ax in axs:
        ax.grid(True)  # Hálóvonalak engedélyezése
        ax.set_facecolor("#F0F0F0")  # Hátterek egységesítése
    fig.patch.set_facecolor("#F0F0F0")  # Az ábra hátterének beállítása
    plt.tight_layout()
    return fig


def save_to_csv(trajectory, config, file_path):
    time = np.hstack(trajectory.dense_time).reshape(-1,1)
    radii = []
    positions = []
    for i in range(config.num_bubbles):
        radii.append(np.hstack(trajectory.dense_states[i]))
        positions.append(np.hstack(trajectory.dense_states[i+config.num_bubbles]))

    time_discrete   = []
    action_discrete = []
    target_positions = []   # Normalized
    for i in range(trajectory.episode_length):
        time_discrete.append([i * config.tspan, (i+1) * config.tspan])
        action_discrete.append([trajectory.actions[i], trajectory.actions[i]])
        target_positions.append([trajectory.observations[i][:config.num_bubbles], trajectory.observations[i][:config.num_bubbles]])

   
    # Save dense output
    file_path = os.path.splitext(file_path)[0] 
    dense_output = file_path + f"_{trajectory.episode_id}_dense.npy"
    radii = np.vstack(radii).T
    positions = np.vstack(positions).T
    dense_data = np.hstack((time, radii, positions))
    np.save(dense_output, dense_data)

    # Save actions
    action_output =  file_path + f"_{trajectory.episode_id}_actions_targets.npy"
    time_discrete = np.hstack(time_discrete).reshape(-1, 1)
    action_discrete = np.vstack(action_discrete)
    target_positions = np.vstack(target_positions)

    discrete_data = np.hstack((time_discrete, action_discrete, target_positions))
    np.save(action_output, discrete_data)



layout = [
    [
        sg.Column([
            [sg.Text("Select Trajectory File:")],
            [sg.Input(key="-FILE-", enable_events=True), sg.FileBrowse()],
            [sg.Listbox(values=[], size=(60, 8), key="-EPISODE_LIST-", enable_events=True)],
            [sg.Button("Save To Csv", size=(23, 1), key="-SAVE-", enable_events=True), sg.Button("Set Parameters", key="-SETUP-", size=(23, 1), enable_events=True)],
            [sg.Text("Acoustic Field:"), sg.Combo(["CONST", "SW_N", "SW_A"], key="-AC-TYPE-", default_value="SW_N", enable_events=True), 
             sg.Text("Num. bubbles:"), sg.Combo(["1", "2"], key="-BUBBLE-NUMBER-", default_value="1", enable_events=True),
             sg.Text("Dimension:"), sg.Combo(["1", "2", "3"], key="-DIM-", default_value="1", enable_events=True)], 
            [sg.Text("Number of Harmonic Components:"), sg.Input(default_text="", key="-NUM_COMPONENTS-", size=(5, 1), enable_events=True)],
            [sg.Text("Excitation Frequencies:")],
            [sg.Text("(kHz)", size=(10,1)), *[sg.Input(f"{(25*(i+1)):.1f}", key=f"-FREQ_{i}-", size=(5, 1), disabled=True, visible=False) for i in range(MAX_COMPONENTS)]],
            [sg.Text("Controlled Pressura Amplitudes (uncrontrolled case min=default):")],
            [sg.Text("Select: ", size=(10, 1)), *[sg.Checkbox(f"{i}", key=f"-PA_{i}-", size=(3, 1), disabled=True, visible=False)  for i in range(MAX_COMPONENTS)]],
            [sg.Text("MIN (bar):", size=(10,1)), *[sg.Input("0.0", key=f"-PA_MIN_{i}-", size=(6, 1), disabled=True, visible=False) for i in range(MAX_COMPONENTS)]],
            [sg.Text("MAX (bar):", size=(10,1)), *[sg.Input("1.0", key=f"-PA_MAX_{i}-", size=(6, 1), disabled=True, visible=False) for i in range(MAX_COMPONENTS)]],
            [sg.Text("Controlled Phase Shift Amplitudes:")],
            [sg.Text("Select: ", size=(10, 1)), *[sg.Checkbox(f"{i}", key=f"-PS_{i}-", size=(3, 1), disabled=True, visible=False) for i in range(MAX_COMPONENTS)]],
            [sg.Text("MIN (π):", size=(10,1)), *[sg.Input("0.0", key=f"-PS_MIN_{i}-", size=(6, 1), disabled=True, visible=False) for i in range(MAX_COMPONENTS)]],
            [sg.Text("MAX (π):", size=(10,1)), *[sg.Input("0.5", key=f"-PS_MAX_{i}-", size=(6, 1), disabled=True, visible=False) for i in range(MAX_COMPONENTS)]],
            [sg.Text("Δt:", size=(7,1)), sg.Input("5.0", key=f"-DT-", size=(5,1)), sg.Text("RES:", size=(6,1)), sg.Input("100", key=f"-DT_RES-", size=(5,1))],
            [sg.Text("ObservationSpace:")],
            [sg.Text("x/λ_MIN:", size=(7,1)), sg.Input("0.0", key=f"-X_MIN-", size=(5,1)), sg.Text("x/λ_MAX:", size=(7,1)), sg.Input("0.25", key=f"-X_MAX-", size=(5,1)), sg.Text("RES:", size=(6,1)), sg.Input("100", key=f"-X_RES-", size=(5,1))],
        ], size=(450, 650), scrollable=False),
        sg.VSeparator(),
        sg.Column([
            [sg.Canvas(key="-CANVAS-")]
        ], size=(800, 650), scrollable=False),
    ],
    [sg.Button("Exit", size=(22,1), pad=(10, 5))]
]




def App():
    window = sg.Window("Trajectory Viewer", layout, finalize=True)
    config = control_configureation()
    acoustic_field = setup_acoustic_field(2, [25.0 * 1e3, 50.0*1e3], "SW_N")

    trajectory_data = None
    figure_canvas_agg = None
    selected_trajectory = None
    file_path = None

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "Exit":
            break

        # ----- File Reading --------
        if event == "-FILE-":
            try:
                file_path = values["-FILE-"]
                trajectory_data = load_trajectories(file_path)
                episodes = [f"ID: {traj.episode_id}, episode_reward: {traj.episode_reward:3f}, episode_length: {traj.episode_length}" for traj in trajectory_data]
                window["-EPISODE_LIST-"].update(episodes)
            except Exception as e:
                sg.popup_error(f"Error loading file: {e}")

        if event == "-SETUP-":
            config.freq = [float(values[f"-FREQ_{i}-"]) for i in range(config.num_harmonic_components)]
            config.pa = [float(values[f"-PA_MIN_{i}-"]) for i in range(config.num_harmonic_components)]
            config.ps = [float(values[f"-PS_MIN_{i}-"]) for i in range(config.num_harmonic_components)]
            config.pa_idx = [i for i in range(config.num_harmonic_components) if values[f"-PA_{i}-"]==True]
            config.ps_idx = [i for i in range(config.num_harmonic_components) if values[f"-PS_{i}-"]==True]
            config.pa_len = len(config.pa_idx)
            config.ps_len = len(config.ps_idx)
            config.xlim = [float(values["-X_MIN-"]), float(values["-X_MAX-"])]
            config.x_res = int(values["-X_RES-"])
            config.tspan = float(values["-DT-"])
            config.t_res = int(values["-DT_RES-"])

            config.setup = True

            # Redraw figure 
            if figure_canvas_agg is not None:
                figure_canvas_agg.get_tk_widget().forget()
                plt.close(fig)
                fig = create_plot(selected_trajectory, config, acoustic_field)
                figure_canvas_agg = draw_figure(window["-CANVAS-"].TKCanvas, fig)

        if event == "-SAVE-":
            if selected_trajectory is not None and config.setup:
                save_to_csv(selected_trajectory, config, values["-FILE-"])


        # --- Change Acoustic Field Parameters ----
        if event == "-NUM_COMPONENTS-":
            try:
                # Get the number of components from the input
                num_components = int(values["-NUM_COMPONENTS-"])
                
                if 0 <= num_components < MAX_COMPONENTS:  # Limit to a maximum of 10 components
                    # Enable or disable input fields based on the number of components
                    config.num_harmonic_components = int(num_components)
                    for i in range(MAX_COMPONENTS):
                        if i < num_components:
                            # Enable the input field and set its background to white
                            window[f"-FREQ_{i}-"].update(disabled=False, visible=True)
                            window[f"-PA_{i}-"].update(disabled=False,   visible=True)
                            window[f"-PA_MIN_{i}-"].update(disabled=False,   visible=True)
                            window[f"-PA_MAX_{i}-"].update(disabled=False,   visible=True)
                            window[f"-PS_MIN_{i}-"].update(disabled=False,   visible=True)
                            window[f"-PS_MAX_{i}-"].update(disabled=False,   visible=True)
                            window[f"-PS_{i}-"].update(disabled=False,   visible=True)
                        else:
                            # Disable the input field and set its background to gray
                            window[f"-FREQ_{i}-"].update(disabled=True, visible=False)
                            window[f"-PA_{i}-"].update(disabled=True,   visible=False)
                            window[f"-PA_MIN_{i}-"].update(disabled=True,   visible=False)
                            window[f"-PA_MAX_{i}-"].update(disabled=True,   visible=False)
                            window[f"-PS_MIN_{i}-"].update(disabled=True,   visible=False)
                            window[f"-PS_MAX_{i}-"].update(disabled=True,   visible=False)
                            window[f"-PS_{i}-"].update(disabled=True,   visible=False)
                else:
                    sg.popup_error("Please enter a number between 0 and 4.")
            except ValueError:
                sg.popup_error("Invalid number. Please enter an integer.")
        
        
        if event == "-BUBBLE-NUMBER-":
            config.num_bubbles = int(values["-BUBBLE-NUMBER-"])

        if event == "-DIM-":
            config.num_bubbles = int(values["-DIM-"])

        if event == "-AC-TYPE-":
            config.num_bubbles = values["-AC-TYPE-"]

        # --- Plot Episode List -----------
        if event == "-EPISODE_LIST-" and trajectory_data and config.setup:
            selected_index = window["-EPISODE_LIST-"].get_indexes()[0]
            selected_trajectory = trajectory_data[selected_index]

            if figure_canvas_agg:
                figure_canvas_agg.get_tk_widget().forget()
                plt.close(fig)

            fig = create_plot(selected_trajectory, config, acoustic_field)
            figure_canvas_agg = draw_figure(window["-CANVAS-"].TKCanvas, fig)


    window.close()


if __name__ == "__main__":
    """
    PA = setup_acoustic_field(2, [25.0 * 1e3, 50.0*1e3], "SW_N")

    t = np.linspace(0, 50, 500)
    x = np.linspace(0, 0.25, 100)

    tt, xx, pp = PA([0.5, 0.1], [0.0, 0.0], x, t)


    #plt.plot(xx[10,:], pp[10,:])
    #plt.show()


    plt.figure(figsize=(10, 6))
    plt.pcolormesh(tt, xx, pp, cmap="viridis", shading="auto")
    plt.colorbar(label="Pressure Amplitude")
    plt.ylabel("Position (x)")
    plt.xlabel("Time (t)")
    plt.title("Pressure Amplitude Colormap")
    plt.show()

    """

    App()
