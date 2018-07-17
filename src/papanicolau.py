import numpy as np
from math import sqrt, pow

from skimage.morphology import dilation, disk
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential

from .display import show_segmentation


def get_cells(image, imshow=False):
    """Threshold image using Otsu automatic threshold.

    Arguments
        image: ndarray
            Image to be thresholded.

        imshow: bool (default False)
            Whether to show thresholded image or not.
    """
    # Segment by threshold
    threshold = threshold_triangle(image)
    bw = image <= threshold

    # Label segmentation
    cells = label(bw)
    cells = __remove_dirt(cells)

    # Find nuclei
    nuclei = __refine(image, cells.copy())
    nuclei[nuclei != 0] = 1
    labeled_nuclei = label(nuclei)

    # Remove segments with high eccentricity
    labeled = __remove_by_eccentricity(cells, labeled_nuclei)
    labeled = relabel_sequential(labeled)[0]

    # Remove small segments
    labeled = __remove_by_area(cells, labeled)
    labeled = relabel_sequential(labeled)[0]

    # Increase segments area
    labeled[labeled != 0] = 1
    labeled = dilation(labeled, disk(5))
    labeled = np.multiply(labeled, bw)
    labeled = label(labeled)

    # Remove small segments again
    labeled = __remove_by_area(cells, labeled)
    labeled = relabel_sequential(labeled)[0]

    # Use other technique to remove small segments
    labeled = __remove_small_segments(labeled)
    labeled = relabel_sequential(labeled)[0]

    # Increase segment area by proximity
    labeled = __relate_pixels(cells, labeled)

    # Remove remaining small segments
    labeled = __remove_small_segments(labeled, 0.2)
    labeled = relabel_sequential(labeled)[0]

    if imshow:
        show_segmentation(image, labeled)

    return labeled


def __remove_dirt(labeled_image):
    props = __regionprops(labeled_image)

    for p in props:
        if p.area > 10:
            continue

        x, y = np.where(labeled_image == p.label)
        labeled_image[x, y] = 0

    return labeled_image


def __mean_area(labeled_image):
    props = __regionprops(labeled_image)
    areas = [p.area for p in props]
    return np.mean(areas)


def __std_area(labeled_image):
    props = __regionprops(labeled_image)
    areas = [p.area for p in props]
    return np.std(areas)


def __mean_eccentricity(labeled_image):
    props = __regionprops(labeled_image)
    areas = [p.eccentricity for p in props]
    return np.mean(areas)


def __std_eccentricity(labeled_image):
    props = __regionprops(labeled_image)
    areas = [p.eccentricity for p in props]
    return np.std(areas)


def __refine(image, labels):
    for p in __regionprops(labels):
        min_row, min_col, max_row, max_col = p.bbox
        subimage = image[min_row:max_row, min_col:max_col]
        sublabels = labels[min_row:max_row, min_col:max_col]

        labels[min_row:max_row, min_col:max_col] = __best_label(subimage,
                                                                sublabels,
                                                                p)

    return __remove_dirt(labels)


def __best_label(image, labels, prop):
    x, y = np.where(labels == prop.label)

    if x.size == 0:
        return labels

    past_labels = []
    # bws = []

    for i in range(20, 100, 5):
        a = i / 100

        threshold = (1 - a) * image[x, y].max() + a * image[x, y].min()

        bw = image <= threshold

        curr_labels = label(bw)
        curr_labels = __remove_dirt(curr_labels)

        past_labels.append(curr_labels)

    max_index, count = __max_labels(past_labels)

    if count == 1:
        return __best_eccentricity(past_labels)
    else:
        best_fit = past_labels[max_index]
        return __refine(image, best_fit)


def __max_labels(past_labels):
    props = [__regionprops(labels) for labels in past_labels]
    number_of_labels = [len(p) for p in props]

    max_index = np.argmax(number_of_labels)
    return max_index, number_of_labels[max_index]


def __best_eccentricity(past_labels):
    props = [__regionprops(labels) for labels in past_labels]
    eccentricities = [p[0].eccentricity if len(p) == 1 else 20 for p in props]

    min_index = np.argmin(eccentricities)
    return past_labels[min_index]


def __regionprops(labels):
    if 1 in labels.shape or len(labels.shape) == 1:
        return []

    return regionprops(labels)


def __remove_by_eccentricity(cells, nuclei):
    cells = cells.copy()
    nuclei = nuclei.copy()
    result = nuclei.copy()

    cells_props = __regionprops(cells)

    for cell in cells_props:
        x, y = np.where(cells == cell.label)

        unique_nuclei_labels = np.unique(nuclei[x, y])[1:]
        unique_n_nuclei_labels = unique_nuclei_labels.size

        if 0 <= unique_n_nuclei_labels <= 1:
            continue

        min_row, min_col, max_row, max_col = cell.bbox
        sub_nuclei = nuclei[min_row:max_row, min_col:max_col]

        outliers = np.setdiff1d(sub_nuclei, unique_nuclei_labels)[1:]
        for outlier in outliers:
            sub_nuclei[sub_nuclei == outlier] = 0

        nuclei_props = np.array(__regionprops(sub_nuclei))

        areas = np.argsort([p.eccentricity for p in nuclei_props])
        biggest_index = areas[0]
        other_nucleus = nuclei_props[biggest_index]

        for nucleus in nuclei_props:
            ratio = (1 - nucleus.eccentricity) / (1 - other_nucleus.eccentricity)

            if ratio < 0.15:
                result[result == nucleus.label] = 0

    return result


def __remove_by_area(cells, nuclei):
    cells = cells.copy()
    nuclei = nuclei.copy()
    result = nuclei.copy()

    cells_props = __regionprops(cells)

    for cell in cells_props:
        x, y = np.where(cells == cell.label)

        unique_nuclei_labels = np.unique(nuclei[x, y])[1:]
        unique_n_nuclei_labels = unique_nuclei_labels.size

        if 0 <= unique_n_nuclei_labels <= 1:
            continue

        min_row, min_col, max_row, max_col = cell.bbox
        sub_nuclei = nuclei[min_row:max_row, min_col:max_col]

        outliers = np.setdiff1d(sub_nuclei, unique_nuclei_labels)[1:]
        for outlier in outliers:
            sub_nuclei[sub_nuclei == outlier] = 0

        nuclei_props = np.array(__regionprops(sub_nuclei))

        areas = np.argsort([p.area for p in nuclei_props])
        biggest_index = areas[-1]
        other_nucleus = nuclei_props[biggest_index]

        for nucleus in nuclei_props:
            ratio = nucleus.area / other_nucleus.area

            if ratio < 0.12:
                result[result == nucleus.label] = 0

    return result


def __relate_pixels(cells, nuclei):
    result = nuclei.copy()
    cells_props = __regionprops(cells)
    nuclei_props = __regionprops(nuclei)

    for cell in cells_props:
        x, y = np.where(cells == cell.label)

        unique_nuclei_labels = np.unique(nuclei[x, y])[1:]
        if unique_nuclei_labels.size == 0:
            continue

        subnuclei_props = __select_nuclei_props(nuclei_props,
                                                unique_nuclei_labels)

        for p, q in zip(x, y):
            if cells[p, q] == 0:
                continue

            new_label = __nearest_label(subnuclei_props, (p, q))
            result[p, q] = new_label

    return result


def __nearest_label(props, p1):
    distances = [__points_dist(p1, __center(p)) for p in props]

    if len(distances) == 0:
        return 0

    index = np.argmin(distances)
    return props[index].label


def __points_dist(p1, p2):
    x, y = p1
    z, w = p2

    a = pow(x - z, 2)
    b = pow(y - w, 2)

    return sqrt(a + b)


def __remove_small_segments(labeled, max_ratio=0.1):
    result = labeled.copy()
    mean_area = __mean_area(labeled)

    for p in __regionprops(labeled):
        ratio = p.area / mean_area

        if ratio < max_ratio:
            x, y = np.where(labeled == p.label)
            result[x, y] = 0

    return result


def __center(prop):
    min_row, min_col, max_row, max_col = prop.bbox
    x = (min_row + max_row) // 2
    y = (min_col + max_col) // 2

    return x, y


def __select_nuclei_props(nuclei_props, unique_nuclei_labels):
    indexes = (unique_nuclei_labels - 1)
    props = []

    for i in indexes:
        props.append(nuclei_props[i])

    return props
