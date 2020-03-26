import os
import shutil
import numpy as np
import cv2 as cv
import random

src_path = "./dataset"
dest_path = "./dest"


def make_dir(path):
  if not os.path.isdir(path):
    os.mkdir(path)


# Detect objects in images
class ObjectDetection:
  def __init__(self, src, dest, threshold):
    self._src = src
    self._dest = dest
    self._threshold = threshold
    make_dir(dest)

  def _do_thresholding(self, file, path):
    bw = cv.imread(file, 0)
    ret, thresh = cv.threshold(bw, self._threshold, 255, cv.THRESH_BINARY)
    cv.imwrite(
      os.path.join(path, f"02-threshold-{self._threshold}.jpg"),
      thresh
    )
    ret, thresh = cv.threshold(bw, self._threshold, 255, cv.THRESH_BINARY_INV)
    cv.imwrite(
      os.path.join(path, f"02-threshold-inverted-{self._threshold}.jpg"),
      thresh
    )
    img = cv.imread(file, 1)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 115, 1)
    cv.imwrite(
      os.path.join(path, f"02-threshold-adaptive-{self._threshold}.jpg"),
      thresh
    )

  def _do_skintoning(self, file, path):
    bw = cv.imread(file, 1)
    hsv = cv.cvtColor(bw, cv.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    hsv_split = np.concatenate((h, s, v), axis=1)
    cv.imwrite(
      os.path.join(path, f"03-hsv.jpg"),
      hsv_split
    )
    ret, min_sat = cv.threshold(s, 40, 255, cv.THRESH_BINARY)
    cv.imwrite(
      os.path.join(path, f"03-min-sat.jpg"),
      min_sat
    )
    ret, max_hue = cv.threshold(s, 15, 255, cv.THRESH_BINARY_INV)
    cv.imwrite(
      os.path.join(path, f"03-max-hue.jpg"),
      max_hue
    )

  def _do_countour(self, file, path):
    img = cv.imread(file, 1)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 115, 1)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)

    img2 = img.copy()
    index = -1
    thickness = 3
    color = (200, 0, 100)

    cv.drawContours(img2, contours, index, color, thickness)
    cv.imwrite(
      os.path.join(path, f"04-contours.jpg"),
      img2
    )

  def _do_canny_edge(self, file, path):
    img = cv.imread(file, 1)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    res, thresh = cv.threshold(hsv[:, :, 0], 25, 255, cv.THRESH_BINARY_INV)
    cv.imwrite(
      os.path.join(path, f"05-canny-thresh.jpg"),
      thresh
    )
    edges = cv.Canny(img, 100, 70)
    cv.imwrite(
      os.path.join(path, f"05-canny.jpg"),
      edges
    )

  def _assign_attributes(self, file, path):
    img = cv.imread(file, 1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 205, 1)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    filtered = []
    for c in contours:
      if cv.contourArea(c) < 1000:
        continue
      filtered.append(c)

    print(len(filtered))

    objects = np.zeros([img.shape[0], img.shape[1], 3], 'uint8')
    for c in filtered:
      col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
      cv.drawContours(objects, [c], -1, col, -1)
      area = cv.contourArea(c)
      p = cv.arcLength(c, True)
      print(area, p)

    cv.imwrite(
      os.path.join(path, f"06-attributes.jpg"),
      objects
    )

  def _do_templates(self, file, path):
    img = cv.imread(file, 1)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    res, thresh = cv.threshold(hsv[:, :, 0], 25, 255, cv.THRESH_BINARY_INV)
    cv.imwrite(
      os.path.join(path, f"05-canny-thresh.jpg"),
      thresh
    )
    edges = cv.Canny(img, 100, 70)
    cv.imwrite(
      os.path.join(path, f"05-canny.jpg"),
      edges
    )

  def produce(self, file):
    dest = os.path.join(self._dest, file)
    make_dir(dest)
    src_original_file = os.path.join(self._src, file)
    dest_original_file = os.path.join(dest, "01-original.jpg")
    if not os.path.isfile(dest_original_file):
      shutil.copyfile(
        src_original_file,
        dest_original_file
      )
    self._do_thresholding(dest_original_file, dest)
    self._do_skintoning(dest_original_file, dest)
    self._do_countour(dest_original_file, dest)
    self._do_canny_edge(dest_original_file, dest)
    self._assign_attributes(dest_original_file, dest)
    self._do_templates(dest_original_file, dest)


detector = ObjectDetection(
  src_path,
  dest_path,
  85
)

detector.produce("10.jpg")
detector.produce("11.jpg")
detector.produce("15.jpg")
detector.produce("20.jpg")
detector.produce("26.jpg")
detector.produce("4.jpg")
detector.produce("41.jpg")
# Create destination directory
# data = enumerate(os.listdir(src_path))
# for idx, val in data:
# 	if ".DS_Store" in val:
# 		continue
# 	detector.produce(val)
# 	print(val)
