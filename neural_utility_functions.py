import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors


def plt_linear(X_train, Y_train, prediction_tf, prediction_np):
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    model_color = '#1f77b4'   # blue
    text_color = '#d62728'    # red

    ax[0].scatter(X_train, Y_train, marker='x', c='black', label="Data Points")
    ax[0].plot(X_train, prediction_tf, c=model_color,
               linewidth=2.5, label="Model output")
    ax[0].text(1.6, 350, r"y=$200 x + 100$",
               fontsize='xx-large', color=text_color)
    ax[0].legend(fontsize='xx-large')
    ax[0].set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
    ax[0].set_xlabel('Size (1000 sqft)', fontsize='xx-large')
    ax[0].set_title("Tensorflow prediction", fontsize='xx-large')

    ax[1].scatter(X_train, Y_train, marker='x', c='black', label="Data Points")
    ax[1].plot(X_train, prediction_np, c=model_color,
               linewidth=2.5, label="Model output")
    ax[1].text(1.6, 350, r"y=$200 x + 100$",
               fontsize='xx-large', color=text_color)
    ax[1].legend(fontsize='xx-large')
    ax[1].set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
    ax[1].set_xlabel('Size (1000 sqft)', fontsize='xx-large')
    ax[1].set_title("Numpy prediction", fontsize='xx-large')
    plt.show()


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    g = 1/(1+np.exp(-z))

    return g


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """ truncates color map """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plt_prob_1d(ax, fwb):
    """ plots a decision boundary but include shading to indicate the probability """
    # setup useful ranges and common linspaces
    x_space = np.linspace(0, 5, 50)
    y_space = np.linspace(0, 1, 50)

    # get probability for x range, extend to y
    z = np.zeros((len(x_space), len(y_space)))
    for i in range(len(x_space)):
        x = np.array([[x_space[i]]])
        z[:, i] = fwb(x)

    cmap = plt.get_cmap('Blues')
    new_cmap = truncate_colormap(cmap, 0.0, 0.5)
    pcm = ax.pcolormesh(x_space, y_space, z,
                        norm=cm.colors.Normalize(vmin=0, vmax=1),
                        cmap=new_cmap, shading='nearest', alpha=0.9)
    ax.figure.colorbar(pcm, ax=ax)


def plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg):
    model_color = '#1f77b4'   # blue
    text_color = '#d62728'    # red
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))

    def layerf(x): return model.predict(x)
    plt_prob_1d(ax[0], layerf)

    ax[0].scatter(X_train[pos], Y_train[pos],
                  marker='x', s=80, c='red', label="y=1")
    ax[0].scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none',
                  edgecolors=model_color, lw=3)

    ax[0].set_ylim(-0.08, 1.1)
    ax[0].set_xlim(-0.5, 5.5)
    ax[0].set_ylabel('y', fontsize=16)
    ax[0].set_xlabel('x', fontsize=16)
    ax[0].set_title('Tensorflow Model', fontsize=20)
    ax[0].legend(fontsize=16)

    def layerf(x): return sigmoid(np.dot(set_w, x.reshape(1, 1)) + set_b)
    plt_prob_1d(ax[1], layerf)

    ax[1].scatter(X_train[pos], Y_train[pos],
                  marker='x', s=80, c='red', label="y=1")
    ax[1].scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none',
                  edgecolors=model_color, lw=3)

    ax[1].set_ylim(-0.08, 1.1)
    ax[1].set_xlim(-0.5, 5.5)
    ax[1].set_ylabel('y', fontsize=16)
    ax[1].set_xlabel('x', fontsize=16)
    ax[1].set_title('Numpy Model', fontsize=20)
    ax[1].legend(fontsize=16)

    fig.text(
        0.1,                # x position
        0.02,               # y position
        "The shading above reflects the output of the sigmoid which varies from 0 to 1.",

        ha='left',
        fontsize=10,
        wrap=True,
        color='darkblue',
        bbox=dict(
            facecolor='lightyellow',
            edgecolor='black',
            boxstyle='round,pad=0.5',
            linewidth=2
        )
    )
    plt.show()
