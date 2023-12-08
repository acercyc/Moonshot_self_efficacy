# %%
import numpy as np
from psychopy import visual, core, event
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------- #
#                                  parameters                                  #
# ---------------------------------------------------------------------------- #
# target
radius = 0.005
vx_max = 0.05
vy_max = 0.05

# environment
line_pos_up = 0.45
line_pos_down = -0.40

# assistance interface
assistance_keyAdd = "right"
assistance_keySub = "left"
assistance_min = 0
assistance_max = 10
Kp, Ki, Kd = 0.1, 0.01, 0.001


# Path
path_n = 3
path_frequency_range = (0.5, 1.3)
nPathSamples = 1000



def generate_path(n, frequency_range, start_y, end_y, nSample=1000, xy_ratio=0.25):
    """
    Generates a path by combining n sine waves with frequencies from the given range.

    :param n: Number of sine waves to combine.
    :param frequency_range: Tuple of (min_frequency, max_frequency) for the sine waves.
    :param sample_rate: Sampling rate in Hz.
    :param duration: Duration of the path in seconds.
    :return: Combined path.
    """

    y = np.linspace(0, 1, int(nSample), endpoint=False)
    x = np.zeros_like(y)

    # assign random weightings to each sine wave
    weightings = np.random.uniform(-1, 1, n)
    weightings /= np.sum(np.abs(weightings))

    frequencys = np.linspace(*frequency_range, n)
    for i, frequency in enumerate(frequencys):
        path = np.sin(2 * np.pi * frequency * y)
        x += path * weightings[i]

    # apply xy ratio
    x *= xy_ratio

    # normalise to start_y and end_y
    y = np.interp(y, (0, 1), (start_y, end_y))

    return x, y


def fn_score_corr(path_x, x, start_y, end_y, nPathSamples):
    """
    Computes the score of the path.

    :param path_x: The path to score.
    :param x: The x values of the mouse trajectory.
    :param y_anchor: The y values of the mouse trajectory.
    :return: The score of the path.
    """
    # intrapolate x
    y_anchor = np.linspace(start_y, end_y, nPathSamples)
    x = np.interp(y_anchor, ys, xs)

    # calculate correlation between path_x and x
    score = np.corrcoef(path_x, x)[0, 1]
    score = (score + 1) / 2
    return score


def fn_score_rms(path_x, x, start_y, end_y, nPathSamples):
    """
    Computes the score of the path.

    :param path_x: The path to score.
    :param x: The x values of the mouse trajectory.
    :param y_anchor: The y values of the mouse trajectory.
    :return: The score of the path.
    """
    # intrapolate x
    y_anchor = np.linspace(start_y, end_y, nPathSamples)
    x = np.interp(y_anchor, ys, xs)

    # calculate correlation between path_x and x
    rms = np.sqrt(np.mean((path_x - x) ** 2))
    score = 1 - rms
    return score


def fn_assistance(mouse_x, mouse_y, path_x, path_y, assistance):
    x_ = np.interp(mouse_y, path_y, path_x)

    # move x based on assistance
    mouse_x = mouse_x + (x_ - mouse_x) * assistance
    return mouse_x


class PIDController:
    def __init__(self, Kp, Ki, Kd, K=1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.K = K
        self.prev_error = 0
        self.sum_error = 0

    def update(self, x, setpoint, dt, K=1):
        self.K = K
        error = setpoint - x
        error_p = error * self.Kp
        error_i = self.sum_error + error * self.Ki * dt
        error_d = self.Kd * (error - self.prev_error) / dt

        control = error_p + error_i + error_d

        self.prev_error = error
        self.sum_error = error_i

        return control * self.K

# ---------------------------------------------------------------------------- #
#                                Initialization                                #
# ---------------------------------------------------------------------------- #
# start and end point
start_x, start_y = 0, line_pos_down
end_x, end_y = 0, line_pos_up
isNewPath = False
assistance = 0

# create path
path_x, path_y = generate_path(
    path_n, path_frequency_range, start_y, end_y, nSample=nPathSamples
)
plt.plot(path_x, path_y)
plt.gca().set_aspect("equal", adjustable="box")



# %%
Kp, Ki, Kd = 1, 0.01, 0.001
PID = PIDController(Kp, Ki, Kd)


# ---------------------------------------------------------------------------- #
#                               Psychopy stimuli                               #
# ---------------------------------------------------------------------------- #
# Create window
win = visual.Window(
    [800, 800], units="height", checkTiming=False, color="black", fullscr=True
)

# up and down boundary lines
line_down = visual.Line(win, start=(-0.5, start_y), end=(0.5, start_y), lineColor="white")
line_up = visual.Line(win, start=(-0.5, end_y), end=(0.5, end_y), lineColor="white")


# Path stimuli
path_true = visual.ShapeStim(
    win, vertices=list(zip(path_x, path_y)), closeShape=False, lineColor=[0.3, 0.3, 0.3]
)

path_self = visual.ShapeStim(
    win, vertices=list(zip(path_x, path_y)), closeShape=False, lineColor=[-1, 1, -1]
)

# target
target = visual.Circle(
    win, radius=radius, fillColor="green", lineColor="None", pos=(start_x, start_y)
)

# assistance rating interface
assistance_rating = visual.RatingScale(
    win,
    low=assistance_min,
    high=assistance_max,
    precision=1,
    size=0.25,
    stretch=4,
    scale=None,
    markerStart=0,
    pos=(0, -0.9),
    textColor="white",
    lineColor="white",
    showValue=False,
    markerColor="white",
    marker="triangle",
    showAccept=False,
    noMouse=True,
)


# Score
text_score = visual.TextStim(
    win, text="score: ", pos=(0, 0.4), color="white", height=0.05
)

# mouse
mouse = event.Mouse(win=win)


# ---------------------------------------------------------------------------- #
#                             Initialise experiment                            #
# ---------------------------------------------------------------------------- #

line_down.draw()
line_up.draw()
text_score.draw()
path_true.draw()
assistance_rating.draw()
target.draw()
win.flip()
mouse.setVisible(0)


# ---------------------------------------------------------------------------- #
#                               Experiment Start                               #
# ---------------------------------------------------------------------------- #
for iTrial in range(100):
    # --------------------------- Trail initialisation --------------------------- #
    # Reset mouse position
    xs = [start_x]
    ys = [start_y]
    target.fillColor = "red"
    target.pos = (start_x, start_y)

    line_down.draw()
    line_up.draw()
    target.draw()
    path_true.draw()
    win.flip()
    core.wait(1)

    target.fillColor = "green"
    line_down.draw()
    line_up.draw()
    target.draw()
    win.flip()

    mouse.setPos((start_x, start_y))
    
    # -------------------------------- Trial start ------------------------------- #
    while True:
        
        # ------------------- Detect and manipulate mouse position ------------------- #
        mouse_x, mouse_y = mouse.getPos()

        # if mouse_y is negative, change it to 0
        if (mouse_y - ys[-1]) <= 0:
            mouse_y = ys[-1]
            mouse_x = xs[-1]
        else:
            # limit y velocity
            if (mouse_y - ys[-1]) > vy_max:
                mouse_y = ys[-1] + vy_max

            # limit x velocity
            if np.abs(mouse_x - xs[-1]) > vx_max:
                mouse_x = xs[-1] + vx_max * np.sign(mouse_x - xs[-1])

            # x boundary
            if np.abs(mouse_x) > 0.5:
                mouse_x = 0.5 * np.sign(mouse_x)

            # add assistance
            # mouse_x = fn_assistance(mouse_x, mouse_y, path_x, path_y, assistance)
            # PID
            dt = mouse_y - ys[-1]
            setpoint = np.interp(mouse_y, path_y, path_x)
            mouse_x_asssistance = PID.update(mouse_x, setpoint, 1/60, assistance)
            mouse_x += mouse_x_asssistance
            
            # # limit x velocity
            # if np.abs(mouse_x - xs[-1]) > vx_max:
            #     mouse_x = xs[-1] + vx_max * np.sign(mouse_x - xs[-1])

            # End trial if mouse reaches the end
            if mouse_y >= end_y:
                break

        # set mouse position
        target.pos = (mouse_x, mouse_y)
        mouse.setPos((mouse_x, mouse_y))

        # ------------------------------ Present stimuli ----------------------------- #
        line_down.draw()
        line_up.draw()
        target.draw()
        assistance_rating.draw()
        win.flip()

        keys = event.getKeys()

        # ------------------------------ Keybaord event ------------------------------ #
        if "q" in keys:
            win.close()
            core.quit()
            break

        if assistance_keyAdd in keys:
            rating_ = assistance_rating.getRating() + 1
            if rating_ > assistance_max:
                rating_ = assistance_max
            assistance_rating.setMarkerPos(rating_)
            isNewPath = True

        elif assistance_keySub in keys:
            rating_ = assistance_rating.getRating() - 1
            if rating_ < assistance_min:
                rating_ = assistance_min
            assistance_rating.setMarkerPos(rating_)
            isNewPath = True
            
        assistance = (assistance_rating.getRating() / assistance_max) ** 3

        xs.append(mouse_x)
        ys.append(mouse_y)

    if "q" in event.getKeys():
        win.close()
        core.quit()
        break

    score = fn_score_rms(path_x, xs, start_y, end_y, nPathSamples)
    text_score.text = f"score: {score*100:.0f}"
    text_score.draw()
    path_true.draw()
    path_self.vertices = list(zip(xs, ys))
    path_self.draw()
    win.flip()
    core.wait(1)

    if isNewPath:
        path_x, path_y = generate_path(
            path_n, path_frequency_range, start_y, end_y, nSample=nPathSamples
        )
        isNewPath = False
        assistance = (assistance_rating.getRating() / assistance_max) ** 3
        path_true.vertices = list(zip(path_x, path_y))

# Close the window
win.close()

q# %%

# %%
