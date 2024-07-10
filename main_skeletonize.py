import numpy as np
import PathCreator
import matplotlib.pyplot as plt
import cv2
from rsf_edges import modelini, get_model_edges

FileName = "B21-166b"
Path0 = "/media/kolad/HardDisk/ThinSection"
PathDir = Path0 + "/" + FileName + "/"
img_edges = cv2.imread(PathDir + "RSF_edges/" + FileName + "_edges_cut.tif")
img = cv2.imread(PathDir + "Picture/" + FileName + ".tif")
sh = img.shape
img = img[
      int(sh[0]/2 - sh[0]/3):int(sh[0]/2 + sh[0]/3),
      int(sh[1]/2 - sh[1]/3):int(sh[1]/2 + sh[1]/3)]

img_edges = img_edges[0:2 ** 9, 0:2 ** 9,:]
result_rsf,a,b = cv2.split(img_edges)
result_rsf = cv2.GaussianBlur(result_rsf,(5,5),cv2.BORDER_DEFAULT)

img = img[0:2 ** 9, 0:2 ** 9,:]

print(img.shape, result_rsf.shape)

def piupiu(alpha1,edges):
      kernel = np.ones((3, 3), np.uint8)
      ret, result_bin = cv2.threshold(result_rsf, alpha1, 255, cv2.THRESH_BINARY)
      result_bin = cv2.erode(result_bin, kernel, iterations=1)
      result_erod = cv2.erode(result_bin, kernel, iterations=1)
      result_erod = cv2.subtract(result_bin, result_erod)
      result_bin = cv2.add(result_bin, edges)
      edges = cv2.ximgproc.thinning(result_bin)
      edges = cv2.subtract(edges, 255 - result_erod)
      return edges

edges_0 = np.zeros((2 ** 9,2 ** 9), np.uint8)
edges_0_1 = edges_0.copy()
edges_0_2 = edges_0.copy()
for i in range(26,255):
      edges = piupiu(i, edges_0_1)
      edges_0_1 = cv2.add(edges_0_1, edges)


fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(2, 2, 1),
      fig.add_subplot(2, 2, 2)]
ax[0].imshow(cv2.merge((result_rsf, result_rsf, result_rsf)))
ax[1].imshow(cv2.merge((edges_0_1, edges_0_1, edges_0_1)))
#ax[2].imshow(cv2.merge((edges_0_1, edges_0_1, edges_0_1)))
#ax[1].imshow(img)

plt.show()


#cv2.add(255 - self.area_bg, self.edges_line)