import os
import shutil
from datetime import datetime

basePath = "/home/david/BlenderProc/SceneNet"
outputDir = "outputTest"

# copy directory for a backup:
shutil.copytree(os.path.join(basePath, outputDir),
                os.path.join(basePath, outputDir + "_backup" + datetime.now().strftime("%Y%m%d-%H%M%S")))

# errorFile = [48,   60,   61,   113,  142,  143,  148,  149,  228,  229,  300,  301,  304,  305, 427, 432,
#              433,  485,  491,  498,  499,  506,  556,  564,  565,  568,  569,  636,  643,  648, 649, 654,
#              655,  714,  715,  716,  717,  737,  780,  781,  782,  783,  841,  842,  854,  858,
#              859,  864,  865,  975,  978,  992,  993,  1070, 1072, 1074, 1075, 1076, 1077, 1080,
#              1081, 1126, 1128, 1129, 1133, 1134, 1135, 1200, 1201, 1206, 1207, 1238, 1239, 1249,
#              1251, 1252, 1271, 1332, 1386, 1387, 1406, 1448, 1449, 1465, 1532, 1580, 1581, 1584,
#              1585, 1594, 1645, 1653, 1662, 1663, 1718, 1719, 1722, 1723, 1733, 1775, 1776, 1778,
#              1779, 1782, 1783, 1852, 1853]
# errorFile = [1870, 1881, 1885, 1886, 1942, 1943, 1945, 1946, 1951, 1952, 1965, 2017, 2029, 2030,
#              2033, 2034, 2051, 2103, 2104, 2115, 2162, 2171, 2172, 2238, 2245, 2246, 2252, 2315,
#              2316, 2319, 2320, 2321, 2381, 2382, 2427, 2442, 2445, 2446, 2513, 2519, 2520, 2525,
#              2526, 2587, 2603, 2604, 2614, 2661, 2665]

# errorFile = [0, 54, 89, 120, 275, 276, 278, 279, 297, 360, 428, 441, 442, 466, 497, 506, 520]
errorFile = []

# delete all files defined as error
for error in errorFile:
    for mode in ['half', 'full']:
        os.remove(os.path.join(basePath, outputDir, mode, 'images', '{:04d}.png'.format(error)))
        os.remove(os.path.join(basePath, outputDir, mode, 'class_annotation', '{:04d}.png'.format(error)))
        os.remove(os.path.join(basePath, outputDir, mode, 'instance_annotation', '{:04d}.png'.format(error)))
        os.remove(os.path.join(basePath, outputDir, mode, 'hdf5', '{:01d}.hdf5'.format(error)))

img_idx = 0
for path in os.listdir(os.path.join(basePath, outputDir, 'half', 'images')):
    if path.endswith(".png"):
        index = path[:-len(".png")]
        if index.isdigit():
            img_idx = max(img_idx, int(index))


# Rename all files
idx = 0
for i in range(img_idx+1):
    if i not in errorFile:
        for mode in ['half', 'full']:
            old_name = os.path.join(basePath, outputDir, mode, 'images', '{:04d}.png'.format(i))
            new_name = os.path.join(basePath, outputDir, mode, 'images', '{:04d}.png'.format(idx))
            os.rename(old_name, new_name)

            old_name = os.path.join(basePath, outputDir, mode, 'class_annotation', '{:04d}.png'.format(i))
            new_name = os.path.join(basePath, outputDir, mode, 'class_annotation', '{:04d}.png'.format(idx))
            os.rename(old_name, new_name)

            old_name = os.path.join(basePath, outputDir, mode, 'instance_annotation', '{:04d}.png'.format(i))
            new_name = os.path.join(basePath, outputDir, mode, 'instance_annotation', '{:04d}.png'.format(idx))
            os.rename(old_name, new_name)

            old_name = os.path.join(basePath, outputDir, mode, 'hdf5', '{:01d}.hdf5'.format(i))
            new_name = os.path.join(basePath, outputDir, mode, 'hdf5', '{:01d}.hdf5'.format(idx))
            os.rename(old_name, new_name)
        idx += 1



