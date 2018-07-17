import warnings
import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.externals import joblib
from src.papanicolau import get_cells
from skimage.measure import regionprops
from skimage.restoration import denoise_tv_chambolle


# Ignore warnings
warnings.filterwarnings("ignore")

# Load image
image_path = 'data/imgs/img2.png'
img = imread(image_path)
img = img[:, :, 0:3]

# Remove noise
denoised = denoise_tv_chambolle(img)

# Get nuclei
gray = rgb2gray(img)

# Segment
cells = get_cells(gray, True)

# Analyze
cells_props = regionprops(cells)
n_rows, n_cols, n_channels = img.shape
model = joblib.load('data/model/knn.pkl')

# Count cells
labels = ['Red', 'Gray', 'Yellow']
counts = [0, 0, 0]

for cell in cells_props:
    x, y = np.where(cells == cell.label)
    mean_intensity = [img[x, y, c].mean() for c in range(n_channels)]

    label = model.predict([mean_intensity])
    label = int(label[0])

    counts[label] += 1

for i in range(3):
    print("{}: {}".format(labels[i], counts[i]))

print("Total: {}".format(sum(counts)))
