import os
import cv2
import xml.etree.ElementTree as ET
from scipy.spatial import distance as dist
import numpy as np
import math

def rotatePoint(xc, yc, xp, yp, theta):

    xoff = xp - xc
    yoff = yp - yc
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return xc + pResx, yc + pResy

def addRotatedShape(cx, cy, w, h, angle):

    p0x, p0y = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)

    p1x, p1y = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)

    p2x, p2y = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)

    p3x, p3y = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)

    points = [[p0x, p0y], [p1x, p1y], [p2x, p2y], [p3x, p3y]]

    return points

def order_points(pts):

    #pts.shape (4,2)
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    if bl[0] == tr[0]:
        if bl[1] > tr[1]:
            bl, tr = tr, bl
    return np.array([tl, tr, br, bl])


class VOCBboxDataset:

    def __init__(self, data_dir):

        id_list_file = data_dir
        self.ids = [id_ for id_ in os.listdir(data_dir) if id_.endswith(".xml")]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):

        id_ = self.ids[i]
        path_xml = os.path.join(self.data_dir,  id_)
        jpg_path = path_xml.replace(".xml", ".jpg")
        txt_path = path_xml.replace(".xml", ".txt")
        img = cv2.imread(jpg_path)
        img_h, img_w, _ = img.shape

        in_file = open(path_xml)
        tree = ET.parse(in_file)
        root = tree.getroot()

        # try:
        with open(txt_path,"w") as file:
            bbox = []
            for obj in root.iter('object'):
                current = list()
                name = obj.find('name').text
                robndbox = obj.find('robndbox')
                cx = float(robndbox.find('cx').text)
                cy = float(robndbox.find('cy').text)
                w = float(robndbox.find('w').text)
                h = float(robndbox.find('h').text)
                angle = float(robndbox.find('angle').text)

                points = addRotatedShape(cx, cy, w, h, angle)
                points = np.array(points).astype(np.int)
                points = order_points(points)
                points[:, 0] = np.clip(points[:, 0], 0, img_w)
                points[:, 1] = np.clip(points[:, 1], 0, img_h)

                list_cordinate = list(points.flatten())
                for i in list_cordinate:
                    file.write(str(i) + ",")
                file.write(name + '\n')
                print(list_cordinate)
                print(name)
                bbox.append(points)
            file.close()



            # for i in bbox:
            #     pts = i.reshape((-1, 1, 2))
            #     cv2.polylines(img, [i], True, (0, 255, 255), 2)
            # cv2.imshow('a', img)
            # cv2.waitKey(0)
            # print(points)
            # except:
            #     print(path_xml)
    __getitem__ = get_example

if __name__ == "__main__":
    voc = VOCBboxDataset(r'D:\ocr\second_ocr\7-28-ocr\normal_data')
    for i in voc:
        print('a')


