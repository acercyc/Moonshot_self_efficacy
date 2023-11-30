# %%
import numpy as np
from psychopy import visual, core, event
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# %%
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


# %%

# Parameters
radius = 0.25
line_pos = 0.45
vx_max = 0.05
vy_max = 0.05
assistance_keyAdd = 'right'
assistance_keySub = 'left'
assistance_min = 0
assistance_max = 10
isNewPath = False

# start point
start_x, start_y = 0, -line_pos
end_x, end_y = 0, line_pos


# create path
nPathSamples = 1000
path_x, path_y = generate_path(3, (0.5, 1.3), start_y, end_y, nSample=nPathSamples)
plt.plot(path_x, path_y)
plt.gca().set_aspect("equal", adjustable="box")
# %%

# Create a window
win = visual.Window(
    [800, 800], units="height", checkTiming=False, color="black", fullscr=True
)

# boundary lines
line_1 = visual.Line(
    win, start=(-0.5, -line_pos), end=(0.5, -line_pos), lineColor="white"
)
line_2 = visual.Line(win, start=(-0.5, end_y), end=(0.5, end_y), lineColor="white")

# target
target = visual.Circle(
    win, radius=0.005, fillColor="green", lineColor="None", pos=(start_x, start_y)
)

# assistance rating
assistance_rating = visual.RatingScale(
    win,
    low=assistance_min,
    high=assistance_max,
    precision=1,
    size=0.25,
    stretch=4,
    scale=None,
    markerStart=0,
    pos=(0, -0.8),
    textColor="white",
    lineColor="white",
    showValue=False,
    markerColor="white",
    marker="triangle",
    showAccept=False,
    noMouse=True
)


# text for score
text_score = visual.TextStim(
    win, text="score: ", pos=(0, 0.4), color="white", height=0.05
)

# Draw the line
line_1.draw()
line_2.draw()
text_score.draw()
assistance_rating.draw()
# Draw the start point
target.draw()


# Show it on the screen
win.flip()

# set assistance
assistance = 0

# record and display mouse position
mouse = event.Mouse(win=win)
mouse.setVisible(0)
for iTrial in range(100):
    # Reset mouse position
    xs = [start_x]
    ys = [start_y]
    target.fillColor = "red"
    target.pos = (start_x, start_y)
    
    line_1.draw()
    line_2.draw()
    target.draw()
    win.flip()
    core.wait(0.5)

    target.fillColor = "green"
    line_1.draw()
    line_2.draw()
    target.draw()
    win.flip()

    mouse.setPos((start_x, start_y))
    while True:
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
            mouse_x = fn_assistance(mouse_x, mouse_y, path_x, path_y, assistance)

            # End trial if mouse reaches the end
            if mouse_y >= end_y:
                break

        # set mouse position
        target.pos = (mouse_x, mouse_y)
        mouse.setPos((mouse_x, mouse_y))

        # Present stimuli
        line_1.draw()
        line_2.draw()
        target.draw()
        assistance_rating.draw()
        win.flip()

        keys = event.getKeys()

        # press 'q' to quit
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
            

        xs.append(mouse_x)
        ys.append(mouse_y)

    if "q" in event.getKeys():
        win.close()
        core.quit()
        break

    score = fn_score_corr(path_x, xs, start_y, end_y, nPathSamples)
    text_score.text = f"score: {score*100:.0f}"
    text_score.draw()
    win.flip()
    core.wait(1)
    
    if isNewPath:
        path_x, path_y = generate_path(3, (0.5, 1.3), start_y, end_y, nSample=nPathSamples)
        isNewPath = False
        assistance = (assistance_rating.getRating()/assistance_max)**3

# Close the window
win.close()


# %%
