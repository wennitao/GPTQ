import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(0.3 * np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(0.5 * np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

path = "/home/v-wentaoni/workspace/amlt/gptq_per_tensor"
tensor_names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj", "mlp.gate_proj"]

benchmarks = ["wikitext2", "ptb", "c4"]

result = {"wikitext2": np.zeros ((32, 7)), "ptb": np.zeros ((32, 7)), "c4": np.zeros((32, 7))}
for job in range (14):
    cur_folder_path = path + "/sparsellm_job_1_" + str (job)
    file_names = os.listdir (cur_folder_path)
    for file_name in file_names:
        if file_name == "logs":
            continue
        tmp = file_name.split ('.')
        layer_id = int (tmp[0].split ('_')[-1])
        tensor_name = tmp[1] + '.' + tmp[2]
        with open (cur_folder_path + "/" + file_name, 'r') as f:
            lines = f.readlines ()
            for i in range (len (lines)):
                lines[i] = lines[i].strip()
                if lines[i] in benchmarks:
                    # print (lines[i], float (lines[i + 1].strip()[5:]))
                    result[lines[i]][layer_id][tensor_names.index (tensor_name)] = float (lines[i + 1].strip()[5:])

layers = []
for i in range (32):
    layers.append ("layer_" + str (i))

for benchmark in benchmarks:
    fig, ax = plt.subplots()

    # print (result[benchmark].shape)
    im, cbar = heatmap(result[benchmark], layers, tensor_names, ax=ax,
                    cmap="YlGn", cbarlabel="PPL")
    texts = annotate_heatmap(im, valfmt="{x:.1f}")

    fig.tight_layout()
    plt.savefig ("plot/" + benchmark + "_3bit.png")