import matplotlib.pyplot as plt

from skimage.color import label2rgb
from math import ceil


def show(images, titles=[]):
    """Display images.

    Arguments:
        images: list
            A list of images.

        titles: list (default [])
            Images' titles. Must be the same length of images.
    """
    n_images = len(images)
    nrows, ncols = __layout(n_images)

    if len(titles) == 0:
            titles = [''] * n_images

    if n_images > 1:
        fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)

        __single_row(images, titles, ax) if nrows == 1 else __many_rows(images,
                                                                        titles,
                                                                        ax,
                                                                        nrows,
                                                                        ncols)
    else:
        plt.imshow(images[0], cmap=plt.cm.gray)
        plt.title(titles[0])
        plt.axis('off')

    plt.tight_layout(pad=0.1)
    plt.show()


def show_segmentation(image, label_image):
    """Show markdowns in labeled image.

    Arguments:
        image: ndarray
            Image used in segmentation.

        label_image: ndarray
            Image with segmentation labels.
    """
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def __layout(n_images):
    if n_images % 3 == 0:
        ncols = 3
    elif n_images % 2 == 0:
        ncols = 2
    else:
        ncols = 3

    nrows = ceil(n_images / ncols)

    return nrows, ncols


def __single_row(images, titles, ax):
    n_images = len(images)

    for i in range(n_images):
        __ax_setting(ax[i], images[i], titles[i])


def __many_rows(images, titles, ax, nrows, ncols):
    k = 0
    n_images = len(images)

    for i in range(nrows):
        for j in range(ncols):
            __ax_setting(ax[i, j], images[k], titles[k])

            k = k + 1
            if k == n_images:
                break


def __ax_setting(ax, image, title):
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_title(title)
    ax.axis('off')
