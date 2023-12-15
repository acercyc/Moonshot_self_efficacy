# %%
import numpy as np
from psychopy import visual, core, gui, event
from datetime import datetime
import os

# ---------------------------------------------------------------------------- #
#                                   switches                                   #
# ---------------------------------------------------------------------------- #
isFullScreen = False # True for full screen, False for windowed mode
isApplyNoise = False

# ---------------------------------------------------------------------------- #
#                        Enter experimental information                        #
# ---------------------------------------------------------------------------- #
info = {
    "Participant ID": "",
    "Number of Block": 3,
    "date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "Practice Mode": False,  # Add this line
}

infoDlg = gui.DlgFromDict(
    dictionary=info,
    title="Input information",
    order=["Participant ID", "Number of Block", "date", "Practice Mode"],  # Add "Mode" here
)

if infoDlg.OK == False:
    core.quit()  # user pressed cancel
    

isPractice = info["Practice Mode"] 
 
# ---------------------------------------------------------------------------- #
#                    Create data folder if it doesn't exist'                   #
# ---------------------------------------------------------------------------- #
if not os.path.exists("data"):
    os.makedirs("data")

# ---------------------------------------------------------------------------- #
#                                  parameters                                  #
# ---------------------------------------------------------------------------- #
nTrial_inBlock = 60
nTrial_practice = 15
nBlock = info["Number of Block"]

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
Kp, Ki, Kd = 1, 0.01, 0.001
assistance_expo_factor = 3

# Path shape
path_n = 3
path_frequency_range = (0.5, 1.3)
nPathSamples = 1000

# Noise
noise_mu = 0
noise_sigma = 0.005


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


# ------------------------------ Score functions ----------------------------- #
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


# -------------------------- score mapping functions ------------------------- #
def fn_map_score_linear(x, x_low=0.8, x_high=1):
    x = np.clip(x, x_low, x_high)
    x = np.interp(x, (x_low, x_high), (0, 1))
    return x


# ------------------------------ Noise function ------------------------------ #
def fn_noise_gaussian(x, mu=0, sigma=0.01):
    return np.random.normal(mu, sigma, x.shape)


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
PID = PIDController(Kp, Ki, Kd)


# ---------------------------------------------------------------------------- #
#                               Psychopy stimuli                               #
# ---------------------------------------------------------------------------- #
# Create window
win = visual.Window(
    [800, 800], units="height", checkTiming=False, color="black", fullscr=isFullScreen
)

# up and down boundary lines
line_down = visual.Line(
    win, start=(-0.5, start_y), end=(0.5, start_y), lineColor="white"
)
line_up = visual.Line(win, start=(-0.5, end_y), end=(0.5, end_y), lineColor="white")


# Path stimuli

# path_true = visual.ShapeStim(
#     win, vertices=None, closeShape=False, lineColor=[0.3, 0.3, 0.3]
# )

# path_self = visual.ShapeStim(
#     win, vertices=None, closeShape=False, lineColor=[-1, 1, -1]
# )
path_x, path_y = generate_path(
    path_n, path_frequency_range, start_y, end_y, nSample=nPathSamples
)


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
    leftKeys=assistance_keySub,
    rightKeys=assistance_keyAdd,
    precision=1,
    size=0.25,
    stretch=4,
    scale=None,
    markerStart=0,
    tickHeight=-1,
    pos=(0, -0.9),
    textColor="white",
    textSize=3,
    lineColor="white",
    showValue=True,
    markerColor="white",
    marker="triangle",
    showAccept=False,
    noMouse=True,
)

# text message
text_message = visual.TextStim(
    win,
    pos=(0, 0),
    color="white",
    height=0.05,
    text=''
)


# Score
text_score = visual.TextStim(
    win, text="score: ", pos=(0, 0.1), color="white", height=0.05
)

# mouse
mouse = event.Mouse(win=win)

# ---------------------------------------------------------------------------- #
#                             Initialise experiment                            #
# ---------------------------------------------------------------------------- #
iTrial = 0
data = {'expInfo': info, 'blockInfo':[], 'trialInfo': []}

# ---------------------------------------------------------------------------- #
#                               Experiment Start                               #
# ---------------------------------------------------------------------------- #
if isPractice:
    welcome_message = f"Welcome to the practice session!\n\nClick mouse button to continue"
else:
    welcome_message = f"Welcome to the experiment!\n\nClick mouse button to continue"

text_message.text = welcome_message
text_message.draw()
win.flip()
mouse.setVisible(0)

# click mouse to start
while True:
    mouse.setPos((0, 0))
    if mouse.getPressed()[0]:
        break
    if "q" in event.getKeys(keyList=["q"]):
        win.close()
        core.quit()
        break

# release mouse press
while True:
    if not mouse.getPressed()[0]:
        break

# ---------------------------------------------------------------------------- #
#                             Initialise block                                 #
# ---------------------------------------------------------------------------- #
for iBlock in range(nBlock):
    # create new path
    path_x, path_y = generate_path(
        path_n, path_frequency_range, start_y, end_y, nSample=nPathSamples
    )
    path_true.vertices = list(zip(path_x, path_y))
    
    # reset assistance
    assistance_rating.setMarkerPos(0)
    assistance = 0
    
    # show block message
    text_message.text = f"[ Block {iBlock+1} ]\n\nClick mouse button to start"
    text_message.draw()
    win.flip()    
    mouse.setVisible(0)


    # save blockInfo
    data['blockInfo'].append({'iBlock': iBlock, 'path_x': path_x, 'path_y': path_y})

    # click mouse to start
    while True:
        mouse.setPos((0, 0))
        if mouse.getPressed()[0]:
            break
        if "q" in event.getKeys(keyList=["q"]):
            win.close()
            core.quit()
            break

    # release mouse press
    while True:
        if not mouse.getPressed()[0]:
            break
    # ---------------------------------------------------------------------------- #
    #                                  Trial loop                                  #
    # ---------------------------------------------------------------------------- #
    if isPractice:
        nTrial = nTrial_practice
    else:
        nTrial = nTrial_inBlock
    
    for iTrial_block in range(nTrial):
        # --------------------------- Trail initialisation --------------------------- #
        # Reset mouse position
        xs = [start_x]
        ys = [start_y]
        ts = []
        target.fillColor = "red"
        target.pos = (start_x, start_y)

        line_down.draw()
        line_up.draw()
        target.draw()
        assistance_rating.draw()
        win.flip()
        
        # --------------------------- assistance operation --------------------------- #
        while True:
            # fix mouse positions
            mouse.setPos((start_x, start_y))
            
            # detect keyboard event
            keys = event.getKeys()
            if "q" in keys:
                win.close()
                core.quit()
                break
            
            # increase or decrease assistance
            if assistance_keyAdd in keys:
                rating_ = assistance_rating.getRating() + 1
                if rating_ > assistance_max:
                    rating_ = assistance_max
                assistance_rating.setMarkerPos(rating_)

            elif assistance_keySub in keys:
                rating_ = assistance_rating.getRating() - 1
                if rating_ < assistance_min:
                    rating_ = assistance_min
                assistance_rating.setMarkerPos(rating_)
                            
            line_down.draw()
            line_up.draw()
            target.draw()
            assistance_rating.draw()
            win.flip()
            
            # click to continue
            if mouse.getPressed()[0]:
                break
        
        # setup assistance
        assistance = (
            assistance_rating.getRating() / assistance_max
        ) ** assistance_expo_factor
        
        # show path and wait for 1 second
        target.fillColor = "yellow"
        line_down.draw()
        line_up.draw()
        target.draw()
        assistance_rating.draw()
        path_true.draw()
        win.flip()
        core.wait(1) 
        
        # go signal
        target.fillColor = "green"
        line_down.draw()
        line_up.draw()
        target.draw()
        assistance_rating.draw()
        ts.append(core.getTime())
        win.flip()
        mouse.setPos((start_x, start_y))
        # ---------------------------------------------------------------------------- #
        #                                  Trial start                                 #
        # ---------------------------------------------------------------------------- #
        while True:
            # get time
            t = core.getTime()

            # ------------------- Detect and manipulate mouse position ------------------- #
            mouse_x, mouse_y = mouse.getPos()

            # if mouse_y shift is negative, change it to 0
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

                # apply noise
                if isApplyNoise:
                    mouse_x += fn_noise_gaussian(mouse_x, noise_mu, noise_sigma)

                # x boundary
                if np.abs(mouse_x) > 0.5:
                    mouse_x = 0.5 * np.sign(mouse_x)

                # PID
                # dt = mouse_y - ys[-1]
                setpoint = np.interp(mouse_y, path_y, path_x)
                # mouse_x_asssistance = PID.update(mouse_x, setpoint, 1/60, assistance)
                mouse_x_asssistance = PID.update(mouse_x, setpoint, t-ts[-1] , assistance)
                mouse_x += mouse_x_asssistance

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

            # ------------------------------ Record data ------------------------------ #
            xs.append(mouse_x)
            ys.append(mouse_y)
            ts.append(t)
            
            # ------------------------------ Keybaord event ------------------------------ #
            if "q" in event.getKeys(keyList=["q"]):
                win.close()
                core.quit()
                break

        # ---------------------------------------------------------------------------- #
        #                                   Trial end                                  #
        # ---------------------------------------------------------------------------- #
        # reset mouse position
        mouse.setPos((start_x, start_y))
        
        # calculate score
        score = fn_score_rms(path_x, xs, start_y, end_y, nPathSamples)
        score_map = fn_map_score_linear(score)
        
        # show score
        text_score.text = f"score: {score_map*100:.0f}"
        text_score.draw()
        
        # show true path
        path_true.draw()
        
        # show path from participant
        path_self.vertices = list(zip(xs, ys))
        path_self.draw()
        win.flip()
        
        # record data
        data_ = {'iBlock': iBlock, 
                 'iTrial': iTrial, 
                 'score': score, 
                 'score_map': score_map, 
                 'assistance': assistance, 
                 'xs': xs, 
                 'ys': ys,
                 'ts': ts,
                 't_start': ts[0],
                 't_end': ts[-1], 
                 't_duration': ts[-1] - ts[0]}
        
        data['trialInfo'].append(data_)
        
        # save data
        if isPractice:
            np.save(f"data/{info['Participant ID']}_{info['date']}_practice.npy", data)
        else:           
            np.save(f"data/{info['Participant ID']}_{info['date']}.npy", data)
        core.wait(1)

# end message
text_message.text=f"End of the experiment!\n\nThank you for your participation!"
text_message.draw()
win.flip()
core.wait(4)


# Close the window
win.close()
core.quit()
