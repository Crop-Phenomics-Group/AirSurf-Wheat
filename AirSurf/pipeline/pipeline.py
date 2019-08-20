# Imports
import numpy as np
import cv2
import math
import csv
from threading import Thread

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import sgd
from keras import backend as K
from keras import utils as np_utils
import keras

import os
import glob
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from skimage.measure import label, regionprops, shannon_entropy
from skimage.feature import greycomatrix, greycoprops
import pickle


# print(os.getcwd())
#%set_env KMP_DUPLICATE_LIB_OK=TRUE # TODO set environment variable from python

class Pipeline(Thread):

    instance = None

    @staticmethod
    def getInstance():
        if Pipeline.instance is None:
            Pipeline.instance = Pipeline()
        return Pipeline.instance

    def __init__(self):
        super().__init__()

    # Process into 9x9 images. It assumes that if they aren't 9x9 they are 11x11
    # because that is the size I first started classifying. In that case the
    # outermost pixels are removed so the center pixel remains the same.
    def preprocess(self, img):
        h, w = img.shape[:2]
        if h is 9 and w is 9:
            return img

        up_img = img[1:10, 1:10]

        return up_img


    # Load images from a base directory that contains two subfolders,
    # one for each of the two classes
    def load_imgs(self, base_path):
        pos_class = "soil"
        neg_class = "not_soil"

        files = []
        ext = ".png"  # May need to make this more comprehensive on a different OS

        data = []
        labels = []

        for (top_dir, dirs, filenames) in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(ext):
                    f = os.path.join(top_dir, filename)
                    img = cv2.imread(f)
                    img = self.preprocess(img)
                    data.append(img)

                    label = f.split(os.path.sep)[-2]
                    labels.append(label)

        return (np.array(data), np.array(labels))


    # Build the CNN
    def build_model(self, width, height, depth, classes):
        model = Sequential()
        shape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            shape = (depth, height, width)

        model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=shape))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation="softmax"))

        return model


    # This extracts 9x9 images from a larger image and returns
    # 4 numpy arrays, the images, an empty array for labels to be added
    # and the coordinates of the top left corner of the images
    def get_small_imgs_from_mosaic(self, img):
        h, w = img.shape[:2]
        step = 6
        size = 9

        step_x = 5
        step_y = 5

        images_to_pred = []
        X = []
        Y = []

        for x in range(0, w - size, step):
            for y in range(0, h - size, step):
                small_img = img[y:y + size, x:x + size]
                small_img = small_img.astype("float") / 255.0
                images_to_pred.append(small_img)
                X.append(x)
                Y.append(y)

        labels = [None] * len(X)

        return (np.array(images_to_pred), np.array(labels), np.array(X), np.array(Y))


    # Courtesy of pyimagesearch.com
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = []
        for i in range(len(pts)):
            s.append(pts[i][0] + pts[i][1])

        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[3] = pts[np.argmin(diff)]
        rect[1] = pts[np.argmax(diff)]

        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        # print("4points")
        # print(rect)

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # print("maxW = " + str(maxWidth))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # print("maxH = " + str(maxHeight))

        dst = np.array([
            [0, 0],
            [0, maxWidth],
            [maxHeight, maxWidth],
            [maxHeight, 0]
        ], dtype="float32")
        # print(rect)
        # print(dst)
        M = cv2.getPerspectiveTransform(rect, dst)
        # print(M)
        M = np.array([
            [1.03645, 0.01255, -84.5],
            [-0.01022, 1.11267, -136.5],
            [-0.00, 0, 1]], dtype="float64")
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  # ,flags=cv2.WARP_INVERSE_MAP)

        return warped

    def get_dots(self, img):
        h, w = img.shape[:2]
        mask = np.zeros((h, w)).astype('uint8')
        # mask[img[:,:]==[0,0,255]] = 255
        for y in range(h):
            for x in range(w):
                if img[y, x, 0] == 0 and img[y, x, 1] == 0 and img[y, x, 2] == 255:
                    mask[y, x] = 255

        # We have a mask with the four red dots white and everything else black.
        labels = label(mask)
        regions = regionprops(labels)

        dots = []
        for region in regions:
            dots.append(region['centroid'])

        cv2.imwrite("mask.png", mask)

        return dots
        # print(dots)

        # warped = four_point_transform(img,dots)

        # cv2.imwrite("warped.png", warped)
        # return warped


    # Combines lines that are too close together, ideally
    # resulting in only 1 line between plots.
    def line_consensus(self, lines):
        lines.sort()
        cons = []
        # print(lines)

        for i in range(len(lines) - 1):
            if lines[i + 1] - lines[i] < 10:
                avg = (lines[i + 1] + lines[i]) / 2
                cons.append(avg)
            # Special case for item first in list
            elif lines[i] - lines[i - 1] >= 10 and i != 0:
                # continue
                cons.append(lines[i])
            elif i == 0:
                cons.append(lines[i])

        # Special case for item last in list
        if lines[-1] - lines[-2] >= 10:
            cons.append(lines[-1])

        return cons


    # Creates a mask image from the x and y coordinates of
    # vertical and horizontal lines, respectively.
    def mask_write(self, shape, ver_cons, hor_cons, name):
        mask = np.zeros(shape)

        # Vertical lines
        for line in ver_cons:
            x = int(line)
            y0 = 0
            y1 = shape[0]
            cv2.line(mask, (x, y0), (x, y1), 255, 1)

        for line in hor_cons:
            y = int(line)
            x0 = 0
            x1 = shape[1]
            cv2.line(mask, (x0, y), (x1, y), 255, 1)

        cv2.imwrite(name, mask)
        return mask


    # Because the fields have small plots that make it so the
    # lines between plots are not uniformly spaced horizontally
    # (on these images, it could work the other way too) I
    # wanted to see if finding the edge lines and then simply
    # equalizing the distance between all the intervening lines
    # would work. It didn't. Either the plots are not exactly
    # the same width, or pixel width is not uniform across the
    # entire image.
    def vert_equalize(self, lines):
        distances = []
        ret_lines = []
        for i in range(len(lines) - 1):
            distances.append(lines[i + 1] - lines[i])

        print(np.mean(distances))
        mean = np.mean(distances)
        least = lines[0]
        most = lines[-1]
        for i in range(len(lines)):
            if i == 0:
                ret_lines.append(least)
            elif i == len(lines) - 1:
                ret_lines.append(most)
            else:
                ret_lines.append(least + mean * i)

        return ret_lines

    def norm_range(self, mat, min_val, max_val):
        width = max_val - min_val

        if width > 0:
            mat -= min_val
            mat /= float(width)

    def vegetative_index(self, img):
        h, w = img.shape[:2]
        if h == 0 and w == 0:
            return
        f_img = img.astype(np.float64) / 255.0  # Normalize the image
        r, g, b = cv2.split(f_img)
        total = r + g + b

        r = np.divide(r, total)
        g = np.divide(g, total)
        b = np.divide(b, total)

        ex_g = 2.0 * g - r - b
        ex_r = 1.4 * r - b
        veg = ex_g - ex_r

        self.norm_range(veg, -2.4, 2.0)

        veg = veg * 255.0
        veg = veg.astype(np.uint8)

        return np.mean(veg)

    def anisotropy(self, img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray.astype('float64')

        grad_x = np.diff(gray, axis=1)
        grad_y = np.diff(gray, axis=0)

        grad_x = grad_x[:h - 1, :]
        grad_y = grad_y[:, :w - 1]

        # square the gradients
        grad_x2 = np.square(grad_x)
        grad_y2 = np.square(grad_y)

        # sum the squared gradients together
        grad_xy2 = np.add(grad_x2, grad_y2)

        # square root the summation matrix
        norm = np.sqrt(grad_xy2)

        # to handle divide by zero case: set the effect of the gradient to 1/255 when too low
        norm[norm < 2] = 255
        gx = np.divide(grad_x, norm)
        gy = np.divide(grad_y, norm)
        nxx = np.multiply(gx, gx)
        nxy = np.multiply(gy, gx)
        nyy = np.multiply(gy, gy)
        xx = np.mean(nxx.flatten())
        xy = np.mean(nxy.flatten())
        yy = np.mean(nyy.flatten())
        # eigenvalues and eigenvector of texture tensor
        m = (xx + yy) / 2.0
        d = (xx - yy) / 2.0
        v = math.sqrt(xy * xy + d * d)
        v1 = m + v
        v2 = m - v
        # direction
        tn = - math.atan((v2 - xx) / float(xy))
        tn = math.degrees(tn)
        # score
        score_n = abs((v1 - v2) / 2.0 / float(m))

        return tn, score_n

    def coverage(self, img):
        h, w = img.shape[:2]
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        luminance = lab_img[:, :, 0]

        mean_lum = np.mean(luminance)
        std_lum = np.std(luminance)

        _, binary = cv2.threshold(luminance, mean_lum - std_lum, 255, cv2.THRESH_BINARY)

        num_pixels = h * w
        num_pix_nonzero = len(np.flatnonzero(binary))

        return float(num_pix_nonzero) / float(num_pixels)

    def green_median(self, img):
        h, w = img.shape[:2]

        g = img[:, :, 1]
        med = np.median(g)

        return med

    def entropy(self, img):
        entro = shannon_entropy(img)
        return entro

    def glcm_entropy(self, img):
        # bw_img = rgb2grey(img)
        # mask = cv2.threshold(bw_img, 127, 255, cv2.THRESH_OTSU)
        # b,g,r = cv2.split(img)
        # ex_g = 2.0 * g - b - r
        # ex_g = ex_g.astype("uint8")
        # ex_g = cv2.cvtColor(ex_g, cv2.COLOR_GRAY2RGB)
        # ex_g = cv2.cvtColor(ex_g, cv2.COLOR_RGB2GRAY)
        # print(ex_g.shape)
        # print(ex_g.dtype)

        mask = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
        # mask = cv2.threshold(ex_g, 127, 255, cv2.THRESH_OTSU)
        # glcm = greycomatrix(mask,10,0, normed=True)
        glcm = greycomatrix(img, [10], [0, np.pi / 2], normed=True, symmetric=True)
        # print(glcm)

        contrast = greycoprops(glcm, 'contrast')[0][0]
        dissimilarity = greycoprops(glcm, 'dissimilarity')[0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0]
        energy = greycoprops(glcm, 'energy')[0]
        correlation = greycoprops(glcm, 'correlation')[0]
        asm = greycoprops(glcm, 'ASM')[0]

        return (contrast, dissimilarity, homogeneity, energy, correlation, asm)

    def remove_edge_effects(self, plot_img):
        h, w = plot_img.shape[:2]
        h_edge = 0
        w_edge = 0
        if h > w:
            h_edge = int(h * 0.15)
            w_edge = int(w * 0.1)
        else:
            h_edge = int(h * 0.1)
            w_edge = int(w * 0.15)

        return plot_img[0 + h_edge:h - h_edge, 0 + w_edge:w - w_edge, :]

    def get_height(self, img):
        if len(img.shape) > 2:
            plot_img = img[:, :, 0]
        else:
            plot_img = img
        area = plot_img.shape[0] * plot_img.shape[1]
        count = 0
        for x in range(plot_img.shape[0]):
            for y in range(plot_img.shape[1]):
                if plot_img[x, y] == 0:
                    count += 1
        plot_img = plot_img.ravel()[np.flatnonzero(plot_img)]

        # Here, if there is not enough data from the point cloud I will return None
        # to indicate that there wasn't enough data for the plot in question
        if len(plot_img) / float(area) < 0.5:
            return None

        height = np.mean(plot_img)
        height = height / 255.0

        return height

    def get_data(self, img):
        plot = self.remove_edge_effects(img)
        veg_i = self.vegetative_index(plot)
        aniso = self.anisotropy(plot)
        cover = self.coverage(plot)
        green = self.green_median(plot)
        entro = self.entropy(plot)
        return (veg_i, aniso[1], cover, green, entro)

    def convert_coords(self, coords, start_shape, target_shape):
        (x, y, w, h, a) = coords

        x = float(x) / float(start_shape[1])
        x = int(x * target_shape[1])
        y = float(y) / float(start_shape[0])
        y = int(y * target_shape[0])
        w = float(w) / float(start_shape[1])
        w = int(w * target_shape[1])
        h = float(h) / float(start_shape[0])
        h = int(h * target_shape[0])
        a = w * h

        return (x, y, w, h, a)

    def print_coords_on_img(self, img,r,c,x,y,w,h):
        font = cv2.FONT_HERSHEY_SIMPLEX
        string = str(r) + ":" + str(c)
        cv2.putText(img,string,(int(x+w/3),int(y+h/2)), font,1, (0,0,255),2,cv2.LINE_AA)
        return img


    def setup(self, img_path, output_dir, hmap_path=None):
        self.img_path = img_path
        self.output_dir = output_dir
        self.hmap_path = hmap_path


    def run(self):
        # Save or load model

        #model.save("models/soil/model_5.h5")
        model = load_model("model_5.h5")
        # Info about model
        model.summary()

        hmap = None

        if self.hmap_path is not None:
            hmap = cv2.imread(self.hmap_path)



        img = cv2.imread(self.img_path)


        if img.shape[2] > 3:
            img = img[:,:,:3]

        img_h, img_w = img.shape[:2]

        hmap_h = 0
        hmap_w = 0

        if hmap is not None:
            hmap_h, hmap_w = hmap.shape[:2]

        # What images should we read in for segmentation
        #img = cv2.imread("/Users/bauera/work/airsurf/air_surf_wheat_cnn/20mb_imgs/ChurchFarm-180613-DFW.png")
        #img = cv2.imread("20mb_pngs/Morley_180611.png")
        (images, labels, x_values, y_values) = self.get_small_imgs_from_mosaic(img)

        print(len(images))

        # Run the model on the small images extracted from the original
        outputs = model.predict(images,verbose=1)

        # Draws the areas classified as soil with >98% confidence on
        # the original image as well as a black and white mask and
        # saves them to disk
        h, w = img.shape[:2]
        size = 9
        step = 6
        out_img = img.copy()
        out_img_bw = np.zeros((h, w))

        for i in range(len(outputs)):
            if outputs[i][1] >= 0.98:
                x = x_values[i]
                y = y_values[i]
                cv2.rectangle(out_img, (x + 1, y + 1), (x + size - 1, y + size - 1), (0, 0, 255), -1)
                cv2.rectangle(out_img_bw, (x + 1, y + 1), (x + size - 1, y + size - 1), 255, -1)
            # out_img_bw[y+1:y+size-1][x+1:x+size-1] = 1
            # out_img_bw[x+1:x+size-1][y+1:y+size-1] = 1

        cv2.imwrite("output.png", out_img)

        out_img_bw = out_img_bw * 255
        out_img_bw.astype("uint8")
        # kernel = np.ones((11,11))
        # out_img_bw =cv2.dilate(out_img_bw,kernel,1)
        # out_img_bw = cv2.morphologyEx(out_img_bw, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("output_bw.png", out_img_bw)

        # Draw all the Hough Lines on the image.
        # Save a black and white mask, and the original
        # with lines drawn on it.
        out = img.copy()
        # cv2.imwrite("test.png",out)
        ### Transition to using Hough lines
        # out_img_bw = cv2.imread("output_bw.png")
        mask = out_img_bw
        hough_bw = np.zeros((h, w))
        # edges = cv2.Canny(out_img,50,150,aperture_size=3)
        lines = cv2.HoughLines(mask.astype("uint8"), 0.1, np.pi / 90, 700)
        # print(lines.shape)
        counts = 0
        for line in lines:
            if line[0][1] > 0.01 and line[0][1] < 1.55:
                continue
            if line[0][1] > 1.58:
                continue
        # print(line)
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 10000 * (-b))
                y1 = int(y0 + 10000 * (a))
                x2 = int(x0 - 10000 * (-b))
                y2 = int(y0 - 10000 * (a))

                cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.line(hough_bw, (x1, y1), (x2, y2), 255, 1)
                counts += 1

        hough_bw = np.bitwise_not(hough_bw.astype("uint8"))
        print(counts)

        cv2.imwrite('houghlines.png', out)
        cv2.imwrite('houghlines_bw.png', hough_bw)

        # Separate the Hough lines into horizontal and vertical lines

        ### I want to combine the nearby lines rather than just making them thicker.
        hor = []
        ver = []

        for line in lines:
            if line[0][1] < 0.01:
                ver.append(line[0, 0])
            elif line[0][1] > 1.55 and line[0][1] < 1.58:
                hor.append(line[0, 0])

        hor.sort()
        ver.sort()
        ver2 = ver
        hor2 = hor
        hor_cons = []
        ver_cons = []

        while len(hor_cons) != len(hor):
            hor_cons = hor
            hor = self.line_consensus(hor)

        while len(ver_cons) != len(ver):
            ver_cons = ver
            ver = self.line_consensus(ver)

        # ver_eq = vert_equalize(ver_cons)
        # Making the horizontal distances equal is not as accurate, because the plots are not all perfectly the same size.
        # print(len(ver_cons))
        # print(len(ver_eq))


        # Write out various images using different sets of lines.
        mask = self.mask_write((h, w), ver_cons, hor_cons, "test0.png")
        mask = np.bitwise_not(mask.astype("uint8"))
        cv2.imwrite("mask2.png", mask)
        # mask_write((h,w),ver_cons,hor_cons,"test.png") # Redundant with test0
        self.mask_write((h, w), ver2, hor2, "test2.png")

        cutout = img.copy()
        output = cv2.bitwise_and(cutout, cutout, mask=mask)
        print(cutout.shape)
        cv2.imwrite("test3.png", output)


        # Write an image that should only include the areas that are in plots, but will also include small plots.
        cutout = img.copy()

        output = cv2.bitwise_and(cutout,cutout,mask=hough_bw)
        print(cutout.shape)
        print(hough_bw.shape)
        cv2.imwrite("cutout.png",output)
        mask = hough_bw

        #b,g,r = cv2.split(img)
        #b = b & hough_bw
        #g = g & hough_bw
        #r = r & hough_bw

        #cutouts = cv2.merge(b,g)
        #cutouts = cv2.merge(cutouts,r)


        ### Extract plots  49x16
        # I was using hard-coded values to see about how big the area of each plot is
        # but it should be detected automatically, using the numbers of rows
        # and columns.
        plots = []

        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            plots.append((x, y, width, height, area))

        areas = []
        for plot in plots:
            areas.append(plot[4])

        area_set = list(set(areas))
        hist = []
        # print(area_set)

        for i in area_set:
            count = 0
            for area in areas:
                if area == i:
                    count += 1

            hist.append(count)

        # plt.bar(hist,area_set)
        # plt.bar(area_set,hist)
        # plt.show()

        # 2016 dimensions
        # rows = 16
        # cols = 49

        # DFW dimensions
        # rows = 34
        # cols = 21

        # Rothamsted dimensions
        # rows = 6
        # cols = 60
        # num_plots = rows * cols
        plot2 = []

        plot2 = [plot for plot in plots if plot[4] > 4000]  # ~5000 for 20mb imgs, 35000 for the 300mb ones
        print(len(plot2))

        plot2 = [plot for plot in plot2 if plot[4] < 80000]
        print(len(plot2))
        # 49*16


        # Create images that are the final mask, as well as the original image
        # with all the non-plot regions removed.
        final_mask = np.zeros(hough_bw.shape)
        for plot in plot2:
            final_mask[plot[1]:plot[1]+plot[3],plot[0]:plot[0]+plot[2]] = 1


        plots_img = cv2.bitwise_and(cutout,cutout,mask=final_mask.astype("uint8"))

        cv2.imwrite("plots.png",plots_img)
        #cv2.imwrite("final_mask.png",final_mask)


        # Check to see if I can overlay plots on heightmap
        # plot is (x,y,w,h,a)

        hmap_plots = []

        if hmap is not None:

            hmap_mask = np.zeros((hmap_h, hmap_w))
            for plot in plot2:
                (x, y, w, h, a) = plot
                x = float(x) / float(img_w)
                x = int(x * hmap_w)
                y = float(y) / float(img_h)
                y = int(y * hmap_h)
                w = float(w) / float(img_w)
                w = int(w * hmap_w)
                h = float(h) / float(img_h)
                h = int(h * hmap_h)
                a = w * h
                hmap_mask[y:y + h, x:x + w] = 1
                hmap_plots.append((x, y, w, h, a))

            hmap_final = cv2.bitwise_and(hmap, hmap, mask=hmap_mask.astype("uint8"))
            cv2.imwrite("heightmap_plots.png", hmap_final)

        # Save each plot image in a folder, with the row and column
        # information in its filename.
        # TODO: This will be changed to saving each plot in a unique
        # folder, and then saving images from different days together.

        dir_name = "dfw_18_07_09"
        # dir_name = "rres_18_05_15"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        if plot2[0][0] > plot2[-1][0]:
            plot2 = list(reversed(plot2))

        grid_row = 1
        grid_col = 1
        raw_y = plot2[0][1]
        plots_with_grids = []

        for plot in plot2:
            if plot[1] != raw_y:
                grid_row += 1
                grid_col = 1
                raw_y = plot[1]
            name = dir_name + "/" + str(grid_row) + "_" + str(grid_col) + ".png"
            cv2.imwrite(name, plots_img[plot[1]:plot[1] + plot[3], plot[0]:plot[0] + plot[2]])
            plots_with_grids.append((grid_row, grid_col, plot[0], plot[1], plot[2], plot[3], plot[4]))

            grid_col += 1

        # This is a clone of the above cell but with hmap_plots instead. Should refactor into
        # a function that takes the plots as an argument

        # dir_name = "../2016_plots_data/08_02"
        # if not os.path.exists(dir_name):
        #    os.mkdir(dir_name)

        if hmap_plots[0][0] > hmap_plots[-1][0]:
            hmap_plots = list(reversed(hmap_plots))

        grid_row = 1
        grid_col = 1
        raw_y = hmap_plots[0][1]
        plots_with_grids = []

        for plot in hmap_plots:
            if plot[1] != raw_y:
                grid_row += 1
                grid_col = 1
                raw_y = plot[1]
            name = dir_name + "/" + str(grid_row) + "_" + str(grid_col) + ".png"
            cv2.imwrite(name, hmap_final[plot[1]:plot[1] + plot[3], plot[0]:plot[0] + plot[2]])
            plots_with_grids.append((grid_row, grid_col, plot[0], plot[1], plot[2], plot[3], plot[4]))

            grid_col += 1

        csv_data = []
        coord_img = hmap.copy()
        print(coord_img.shape)
        for plot in plots_with_grids:
            (r, c, x, y, w, h, a) = plot
            (x, y, w, h, a) = self.convert_coords((x, y, w, h, a), img.shape[:2],
                                             hmap.shape[:2])  # Uncomment when using hmap
            coord_img = self.print_coords_on_img(coord_img, r, c, x, y, w, h)

            plot_img = hmap[y:y + h, x:x + w]
            plot_img = self.remove_edge_effects(plot_img)
            veg_i, aniso, cover, green, entro = self.get_data(plot_img)

            # (r,c,x,y,w,h,a) = plot
            # (x,y,w,h,a) = convert_coords((x,y,w,h,a),img.shape[:2],hmap.shape[:2])
            # hplot = remove_edge_effects(hmap[y:y+h,x:x+w])
            # height = get_height(hplot)

            # For initial good img
            # csv_data.append((r,c,veg_i, aniso, cover, green, entro, height))
            # For image with grid overlaid
            csv_data.append((r, c, veg_i, aniso, cover, green, entro))
            # For heatmap with grid overlaid
            # csv_data.append((r,c,height))

        print(len(csv_data))
        # cv2.imwrite("index.png",coord_img)
        # pickle_name = "180515rres.pickle"
        # pickle.dump(plot2,open(pickle_name,'wb'))

        plot0 = plots_with_grids[21]
        (r, c, x, y, w, h, a) = plot0
        (x, y, w, h, a) = self.convert_coords((x, y, w, h, a), img.shape[:2], hmap.shape[:2])
        hplot = self.remove_edge_effects(hmap[y:y + h, x:x + w])
        height = self.get_height(hplot)
        cv2.imwrite("test_plot.png", hmap[y:y + h, x:x + w])
        # print(height)
        # with open('test.csv','w') as csvfile:
        #    data_w = csv.writer(csvfile)
        #    for row in hplot:
        #        data_w.writerow(row)

        # dir_name = "dfw_18_07_23"
        dir_name = "dfw_18_07_09_h"
        with open(dir_name + ".csv", 'w') as csvfile:
            data_w = csv.writer(csvfile)
            # Row IDX - row
            # Column IDX - column
            # Veg Greenness IDX - vegetative index
            # Canopy Orientation - Isotropy (high numbers indicates likely lodging)
            # Canopy Structure - Shannon Entropy score
            # Greenness Reading - Median value of green channel
            # Relative height - after further work we can attempt to give an absolute value
            data_w.writerow(
                ['Row IDX', 'Column IDX', 'Veg. Greenness IDX', 'Canopy Orientation', 'Coverage', 'Greenness Reading',
                 'Canopy Structure', 'Relative Height'])
            for row in csv_data:
                string = []
                for item in row:
                    if item is not None:
                        string.append("%.3f" % item if not float(item).is_integer() else item)
                    else:
                        string.append("N/A")
                data_w.writerow(string)


