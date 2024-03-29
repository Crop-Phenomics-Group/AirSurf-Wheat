# Imports
import numpy as np
import cv2
import math
import csv
from threading import Thread
import re
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.models import load_model
from keras import backend as K
import os
from skimage.measure import label, regionprops, shannon_entropy
from skimage.feature import greycomatrix, greycoprops

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [0, maxWidth],
            [maxHeight, maxWidth],
            [maxHeight, 0]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
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

    # Normalize an array between 0 and 1
    def norm_range(self, mat, min_val, max_val):
        width = max_val - min_val

        if width > 0:
            mat -= min_val
            mat /= float(width)

    # Calculate the mean vegetative index
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

    # Calculate anisotropy, courtesy of Chris Applegate
    def anisotropy(self, img):
        h, w = img.shape[:2]
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = img
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

    # Return the average canopy coverage across the plot
    def coverage(self, img):
        h, w = img.shape[:2]
        thresh = threshold_otsu(img)
        binary_img = img > thresh

        num_px = h * w
        num_px_nonzero = len(np.flatnonzero(binary_img))

        return float(num_px_nonzero) / float(num_px)

    # Return median greenness of a plot
    def green_median(self, img):
        h, w = img.shape[:2]

        g = img[:, :, 1]
        med = np.median(g)

        return med

    # Return Shannon Entropy of a plot
    def entropy(self, img):
        entro = shannon_entropy(img)
        return entro

    # Calculate GLCM traits on the texture image
    def glcm_entropy(self, img):
        img_max = np.max(img)
        img_min = np.min(img)
        img_copy = img.copy()
        self.norm_range(img_copy,img_min,img_max)
        norm_img = img_as_ubyte(img_copy)
        glcm = greycomatrix(norm_img, [10], [0, np.pi / 2], normed=True, symmetric=True)

        contrast = greycoprops(glcm, 'contrast')[0][0]
        dissimilarity = greycoprops(glcm, 'dissimilarity')[0][0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0]
        energy = greycoprops(glcm, 'energy')[0]
        correlation = greycoprops(glcm, 'correlation')[0][0]
        asm = greycoprops(glcm, 'ASM')[0]

        return (correlation, dissimilarity)

    # Remove the images edge effects
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

    # Calculate median height from a heatmap
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

    # Call the various data extraction methods and return it all in one tuple
    def get_data(self, img):
        f_img = img.astype(np.float64) / 255.0  # Normalize the image
        blue, green, red = cv2.split(f_img)

        ex_g = 2.0 * green - red - blue
        ex_r = 1.4 * red - blue
        veg = ex_g - ex_r

        vari = (green - red) / (green + red - blue)

        self.norm_range(veg, -2.4, 2.0)

        med_g = np.median(green)
        mean_g = np.mean(green)
        med_ex_g = np.median(ex_g)
        mean_ex_g = np.mean(ex_g)
        med_ex_r = np.median(ex_r)
        mean_ex_r = np.mean(ex_r)
        med_veg = np.median(veg)
        mean_veg = np.mean(veg)
        med_vari = np.median(vari)
        mean_vari = np.mean(vari)

        cover = self.coverage(ex_g)
        aniso_direction, aniso = self.anisotropy(ex_g)
        entro = self.entropy(ex_g)
        # ex_g[ex_g > 1] = 1
        corr, diss = self.glcm_entropy(ex_g)

        return (med_g, mean_g, med_ex_g, mean_ex_g, med_ex_r, mean_ex_r, med_veg, mean_veg, med_vari, mean_vari,
                cover, aniso_direction, aniso, entro, corr, diss)

    # Convert coordinates between 2 differently-sized images/arrays so they appear in the same place
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

        return x, y, w, h, a

    # Overlay the plot row and column over the image for manual verification of each plot
    def print_coords_on_img(self, img, r, c, x, y, w, h):
        font = cv2.FONT_HERSHEY_SIMPLEX
        string = str(r) + ":" + str(c)
        cv2.putText(img,string,(int(x+w/3),int(y+h/2)), font,1, (0,0,255),2,cv2.LINE_AA)
        return img

    # Perform some initial path maintenance
    def setup(self, img_path, output_dir, hmap_path=None):
        self.img_path = img_path
        self.output_dir = output_dir
        self.hmap_path = hmap_path

    # Perform all the analysis for a single image, with or without a heightmap
    def single_img_analysis(self):
        self.img = cv2.imread(self.img_path)
        self.hmap = None
        if self.hmap_path is not None:
            self.hmap = cv2.imread(self.hmap_path)
        if self.img.shape[2] > 3:
            self.img = self.img[:,:,:3]

        self.img_h, self.img_w = self.img.shape[:2]

        self.hmap_h = 0
        self.hmap_w = 0

        if self.hmap is not None:
            self.hmap_h, self.hmap_w = self.hmap.shape[:2]


        (images, labels, x_values, y_values) = self.get_small_imgs_from_mosaic(self.img)

        # Run the model on the small images extracted from the original
        self.outputs = self.model.predict(images)

        size = 9
        out_img = self.img.copy()
        out_img_bw = np.zeros((self.img_h, self.img_w))

        for i in range(len(self.outputs)):
            if self.outputs[i][1] >= 0.98:
                x = x_values[i]
                y = y_values[i]
                cv2.rectangle(out_img, (x + 1, y + 1), (x + size - 1, y + size - 1), (0, 0, 255), -1)
                cv2.rectangle(out_img_bw, (x + 1, y + 1), (x + size - 1, y + size - 1), 255, -1)

        cv2.imwrite(os.path.join(self.output_dir,"output.png"), out_img)

        out_img_bw = out_img_bw * 255
        out_img_bw.astype("uint8")
        cv2.imwrite(os.path.join(self.output_dir,"output_bw.png"), out_img_bw)

        # Draw all the Hough Lines on the image.
        # Save a black and white mask, and the original
        # with lines drawn on it.
        out = self.img.copy()

        ### Transition to using Hough lines
        mask = out_img_bw
        hough_bw = np.zeros((self.img_h, self.img_w))
        lines = cv2.HoughLines(mask.astype("uint8"), 0.1, np.pi / 90, 700)
        counts = 0
        for line in lines:
            if line[0][1] > 0.01 and line[0][1] < 1.55:
                continue
            if line[0][1] > 1.58:
                continue
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 10000 * (-b))
                y1 = int(y0 + 10000 * a)
                x2 = int(x0 - 10000 * (-b))
                y2 = int(y0 - 10000 * a)

                cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.line(hough_bw, (x1, y1), (x2, y2), 255, 1)
                counts += 1

        hough_bw = np.bitwise_not(hough_bw.astype("uint8"))

        cv2.imwrite(os.path.join(self.output_dir,'houghlines.png'), out)
        cv2.imwrite(os.path.join(self.output_dir,'houghlines_bw.png'), hough_bw)

        # Separate the Hough lines into horizontal and vertical lines
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

        # Write out various images using different sets of lines.
        mask = self.mask_write((self.img_h, self.img_w), ver_cons, hor_cons, os.path.join(self.output_dir,"test0.png"))
        mask = np.bitwise_not(mask.astype("uint8"))
        cv2.imwrite(os.path.join(self.output_dir,"mask2.png"), mask)
        self.mask_write((self.img_h, self.img_w), ver2, hor2, os.path.join(self.output_dir,"test2.png"))

        cutout = self.img.copy()
        output = cv2.bitwise_and(cutout, cutout, mask=mask)
        cv2.imwrite(os.path.join(self.output_dir,"test3.png"), output)

        # Write an image that should only include the areas that are in plots, but will also include small plots.
        cutout = self.img.copy()

        output = cv2.bitwise_and(cutout,cutout,mask=hough_bw)
        cv2.imwrite(os.path.join(self.output_dir,"cutout.png"),output)
        # mask = hough_bw

        # Extract plots
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

        for i in area_set:
            count = 0
            for area in areas:
                if area == i:
                    count += 1

            hist.append(count)

        plot2 = [plot for plot in plots if plot[4] > 10000]  # ~5000 for 20mb imgs, 35000 for the 300mb ones

        plot2 = [plot for plot in plot2 if plot[4] < 80000]

        # Create images that are the final mask, as well as the original image
        # with all the non-plot regions removed.
        final_mask = np.zeros(hough_bw.shape)
        for plot in plot2:
            final_mask[plot[1]:plot[1]+plot[3],plot[0]:plot[0]+plot[2]] = 1

        plots_img = cv2.bitwise_and(cutout,cutout,mask=final_mask.astype("uint8"))

        cv2.imwrite(os.path.join(self.output_dir,"plots.png"),plots_img)

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
            #name = dir_name + "/" + str(grid_row) + "_" + str(grid_col) + ".png"
            #cv2.imwrite(name, plots_img[plot[1]:plot[1] + plot[3], plot[0]:plot[0] + plot[2]])
            plots_with_grids.append((grid_row, grid_col, plot[0], plot[1], plot[2], plot[3], plot[4]))

            grid_col += 1

        self.csv_data = []
        for plot in plots_with_grids:
            (r, c, x, y, w, h, a) = plot

            plot_img = self.img[y:y + h, x:x + w]
            plot_img = self.remove_edge_effects(plot_img)
            (med_g, mean_g, med_ex_g, mean_ex_g, med_ex_r, mean_ex_r, med_veg, mean_veg, med_vari, mean_vari, cover, aniso_direction, aniso, entro, corr, diss) = self.get_data(plot_img)

            if self.hmap_path is None:
                self.csv_data.append((r, c, med_g, mean_g, med_ex_g, mean_ex_g, med_ex_r, mean_ex_r, med_veg, mean_veg,
                                      med_vari, mean_vari, cover, aniso_direction, aniso, entro, corr, diss))

            else:
                (x,y,w,h,a) = self.convert_coords((x,y,w,h,a),self.img.shape[:2],self.hmap.shape[:2])
                hplot = self.remove_edge_effects(self.hmap[y:y+h,x:x+w])
                height = self.get_height(hplot)
                self.csv_data.append((r, c, med_g, mean_g, med_ex_g, mean_ex_g, med_ex_r, mean_ex_r, med_veg, mean_veg,
                                      med_vari, mean_vari, cover, aniso_direction, aniso, entro, corr, diss, height))

        with open("data.csv", 'w') as csvfile:
            data_w = csv.writer(csvfile)
            if self.hmap_path is None:
                data_w.writerow(
                    ['Row IDX', 'Column IDX', 'Median Greenness', 'Mean Greenness', 'Median Ex_Green', 'Mean Ex_Green',
                     'Median Ex_Red', 'Mean Ex_Red', 'Median Veg IDX', 'Mean Veg IDX', 'Median VARI', 'Mean VARI',
                     'Canopy Coverage', 'Anisotropy Orientation', 'Anisotropy Score', 'Shannon Entropy',
                     'GLCM Correlation', 'GLCM Dissimilarity'])
            else:
                data_w.writerow(['Row IDX','Column IDX','Median Greenness','Mean Greenness','Median Ex_Green','Mean Ex_Green',
                                 'Median Ex_Red','Mean Ex_Red','Median Veg IDX','Mean Veg IDX','Median VARI','Mean VARI',
                                 'Canopy Coverage','Anisotropy Orientation','Anisotropy Score','Shannon Entropy',
                                 'GLCM Correlation','GLCM Dissimilarity','Relative Height'])
            for row in self.csv_data:
                string = []
                for item in row:
                    if item is not None:
                        string.append("%.3f" % item if not float(item).is_integer() else item)
                    else:
                        string.append("N/A")
                data_w.writerow(string)

    # Return true if the directory has no letters, which assumes then that
    # it is a date and valid directory of images
    def dir_is_date(self, dir):
        if re.search('[a-z]',dir) is None:
            return True
        else:
            return False

    # Search for and identify each relevant image in a subdirectory for
    # orthomosaic, heightmap and overhead images
    def get_imgs_from_dates(self):
        dir_structure = os.walk(self.parent_dir)
        for info in dir_structure:
            dirs = info[1]
            break

        self.date_dirs = []
        for dir in dirs:
            if self.dir_is_date(dir) is True:
                self.date_dirs.append(dir)
        self.date_dirs.sort()

        self.orthos = []
        self.heightmaps = []
        self.overviews = []

        for date in self.date_dirs:
            dir_structure = os.walk(os.path.join(self.parent_dir, date))
            for info in dir_structure:
                overview = False
                height = False
                ortho = False
                for img in info[2]:
                    if re.search('[Oo]verview', img) is not None and not overview:
                        self.overviews.append(os.path.join(self.parent_dir, date, img))
                        overview = True
                    elif re.search('[Hh]eight', img) is not None and not height:
                        self.heightmaps.append(os.path.join(self.parent_dir, date, img))
                        height = True
                    elif re.search('[Ss]mall', img) is not None and not ortho:
                        self.orthos.append(os.path.join(self.parent_dir, date, img))
                        ortho = True

                if overview is False:
                    self.overviews.append(None)
                if height is False:
                    self.heightmaps.append(None)
                if ortho is False:
                    self.orthos.append(None)

    # Extract data for a single date while in the series analysis
    def get_data_series_single(self, i):
        ortho = None
        overview = None
        heightmap = None
        csv_data_single = []
        if self.orthos[i] is not None:
            ortho = cv2.imread(self.orthos[i])
        if self.overviews[i] is not None:
            overview = cv2.imread(self.overviews[i])
        if self.heightmaps[i] is not None:
            heightmap = cv2.imread(self.overviews[i])

        # There isn't a proper set of data for this date
        if ortho is None and overview is None:
            return None

        if ortho is not None:
            img_h,img_w = ortho.shape[:2]
            head,tail = os.path.split(self.orthos[i])
            date = os.path.basename(head)
            # print(date)

            for plot in self.plot_coords_grids:
                (r, c, orig_x, orig_y, orig_w, orig_h, orig_a) = plot
                (x,y,w,h,a) = self.convert_coords((orig_x,orig_y,orig_w,orig_h,orig_a),(self.seg_h,self.seg_w),(img_h,img_w))

                plot_img = ortho[y:y + h, x:x + w]
                plot_img = self.remove_edge_effects(plot_img)
                med_g, mean_g, med_ex_g, mean_ex_g, med_ex_r, mean_ex_r, med_veg, mean_veg, med_vari, mean_vari, cover, aniso_direction, aniso, entro, corr, diss = self.get_data(plot_img)

                if heightmap is not None:
                    (x,y,w,h,a) = self.convert_coords((orig_x,orig_y,orig_w,orig_h,orig_a),(self.seg_h,self.seg_w),heightmap.shape[:2])
                    hplot = self.remove_edge_effects(heightmap[y:y+h,x:x+w])
                    height = self.get_height(hplot)
                    csv_data_single.append((r, c, med_g, mean_g, med_ex_g, mean_ex_g, med_ex_r, mean_ex_r, med_veg, mean_veg,
                                    med_vari, mean_vari, cover, aniso_direction, aniso, entro, corr, diss, height))

                else:
                    csv_data_single.append((r, c, med_g, mean_g, med_ex_g, mean_ex_g, med_ex_r, mean_ex_r, med_veg, mean_veg,
                                    med_vari, mean_vari, cover, aniso_direction, aniso, entro, corr, diss))


            csv_file = os.path.join(self.csv_date_path, date)
            csv_file = csv_file + ".csv"

            with open(csv_file, 'w') as csvfile:
                data_w = csv.writer(csvfile)
                if heightmap is None:
                    data_w.writerow(
                        ['Row IDX', 'Column IDX', 'Median Greenness', 'Mean Greenness', 'Median Ex_Green', 'Mean Ex_Green',
                         'Median Ex_Red', 'Mean Ex_Red', 'Median Veg IDX', 'Mean Veg IDX', 'Median VARI', 'Mean VARI',
                         'Canopy Coverage', 'Anisotropy Orientation', 'Anisotropy Score', 'Shannon Entropy',
                         'GLCM Correlation', 'GLCM Dissimilarity', 'Relative Height'])
                else:
                    data_w.writerow(['Row IDX','Column IDX','Median Greenness','Mean Greenness','Median Ex_Green','Mean Ex_Green',
                                     'Median Ex_Red','Mean Ex_Red','Median Veg IDX','Mean Veg IDX','Median VARI','Mean VARI',
                                     'Canopy Coverage','Anisotropy Orientation','Anisotropy Score','Shannon Entropy',
                                     'GLCM Correlation','GLCM Dissimilarity','Relative Height'])
                for row in csv_data_single:
                    string = []
                    for item in row:
                        if item is not None:
                            string.append("%.3f" % item if not float(item).is_integer() else item)
                        else:
                            string.append("N/A")
                    data_w.writerow(string)

    # Use pandas to transform the data into more useable values, allowing traits to be
    # tracked across the whole season
    def perform_data_transformation(self):
        dir_structure = os.walk(self.csv_date_path)
        for info in dir_structure:
            files = info[2]
            break

        csvs = []
        dates = []

        files.sort()

        for file in files:
            if file.startswith("."):
                continue
            csvs.append(pd.read_csv(os.path.join(self.csv_date_path,file)))
            dates.append(file.split(".")[0])

        cols = csvs[0].columns
        traits = []
        for c in cols:
            if c == 'Row IDX' or c == 'Column IDX':
                continue
            else:
                traits.append(c)
        date = csvs[0].iloc[:, 0:2]
        date = pd.concat([date, csvs[0].iloc[:, 3]], axis=1)

        date_base = csvs[0].iloc[:, 0:2]
        count = 0
        trait_csvs = []
        col_names = ['Row IDX', 'Column IDX']
        for t in range(len(traits)):
            t_csv = date_base.copy()
            for i in range(len(dates)):
                t_csv = pd.concat([t_csv, csvs[i].iloc[:, t + 2]], axis=1)
                if t == 0:
                    col_names.append(dates[i])

            t_csv.columns = col_names
            trait_csvs.append(t_csv)
        print(count)

        print(len(trait_csvs))
        print(len(traits))

        trait_csv_path = os.path.join(self.csv_path, "trait_csvs")
        rel_trait_path = os.path.join(self.csv_path, "relative_trait_csvs")
        if not os.path.exists(trait_csv_path):
            os.mkdir(trait_csv_path)
        if not os.path.exists(rel_trait_path):
            os.mkdir(rel_trait_path)


        for i in range(len(trait_csvs)):
            trait_csvs[i].to_csv(path_or_buf=os.path.join(trait_csv_path,traits[i] + ".csv"),
                                 index=False, float_format='%.3f')

        rel_trait_csvs = []
        for csv in trait_csvs:
            csv_copy = csv.copy()
            for c in range(14, 2, -1):
                csv_copy.iloc[:, c] = csv_copy.iloc[:, c] - csv_copy.iloc[:, c - 1]
            csv_copy.iloc[:, 2] = 0
            rel_trait_csvs.append(csv_copy)

        for i in range(len(rel_trait_csvs)):
            rel_trait_csvs[i].to_csv(
                path_or_buf=os.path.join(rel_trait_path,traits[i] + ".csv"),
                index=False, float_format='%.3f')

    # Perform series analysis for many dates
    def series_analysis(self):
        self.get_imgs_from_dates()

        dir_structure = os.walk(self.seg_path)
        for info in dir_structure:
            files = info[2]
            break

        self.seg_img = None

        for file in files:
            if re.search('[Ss]mall', file) is not None:
                self.seg_img = cv2.imread(os.path.join(self.seg_path,file))

        if self.seg_img is None:
            exit(4)

        if self.seg_img.shape[2] > 3:
            self.seg_img = self.seg_img[:,:,:3]

        self.seg_h, self.seg_w = self.seg_img.shape[:2]

        (images, labels, x_values, y_values) = self.get_small_imgs_from_mosaic(self.seg_img)

        print("Beginning segmentation")

        self.outputs = self.model.predict(images)

        size = 9
        out_img = self.seg_img.copy()
        out_img_bw = np.zeros((self.seg_h, self.seg_w))

        for i in range(len(self.outputs)):
            if self.outputs[i][1] >= 0.98:
                x = x_values[i]
                y = y_values[i]
                cv2.rectangle(out_img, (x + 1, y + 1), (x + size - 1, y + size - 1), (0, 0, 255), -1)
                cv2.rectangle(out_img_bw, (x + 1, y + 1), (x + size - 1, y + size - 1), 255, -1)

        cv2.imwrite("output.png", out_img)
        out_img_bw = out_img_bw * 255
        out_img_bw.astype("uint8")
        cv2.imwrite("output_bw.png", out_img_bw)

        # Draw all the Hough Lines on the image.
        # Save a black and white mask, and the original
        # with lines drawn on it.
        out = self.seg_img.copy()
        # cv2.imwrite("test.png",out)
        ### Transition to using Hough lines
        # out_img_bw = cv2.imread("output_bw.png")
        mask = out_img_bw
        hough_bw = np.zeros((self.seg_h, self.seg_w))
        lines = cv2.HoughLines(mask.astype("uint8"), 0.1, np.pi / 90, 1000) # TODO: Add dynamic line length for
        # TODO: Hough lines, difficult because two identically-sized images can have wildly different outcomes
        # TODO: with the same parameters.
        counts = 0
        for line in lines:
            # 0 and pi/2 represent horizontal and vertical lines, so skip anything that isn't one of these two
            if line[0][1] > 0.01 and line[0][1] < 1.55:
                continue
            if line[0][1] > 1.58:
                continue
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 10000 * (-b))
                y1 = int(y0 + 10000 * a)
                x2 = int(x0 - 10000 * (-b))
                y2 = int(y0 - 10000 * a)

                cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.line(hough_bw, (x1, y1), (x2, y2), 255, 1)
                counts += 1

        hough_bw = np.bitwise_not(hough_bw.astype("uint8"))

        cv2.imwrite('houghlines.png', out)
        cv2.imwrite('houghlines_bw.png', hough_bw)

        # Separate the Hough lines into horizontal and vertical lines

        # I want to combine the nearby lines rather than just making them thicker.
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

        # Write out various images using different sets of lines.
        mask = self.mask_write((self.seg_h, self.seg_w), ver_cons, hor_cons, "test0.png")
        mask = np.bitwise_not(mask.astype("uint8"))
        # cv2.imwrite("mask2.png", mask)
        # self.mask_write((self.seg_h, self.seg_w), ver2, hor2, "test2.png")

        # cutout = self.seg_img.copy()
        # output = cv2.bitwise_and(cutout, cutout, mask=mask)
        # print(cutout.shape)
        # cv2.imwrite("test3.png", output)

        # Write an image that should only include the areas that are in plots, but will also include small plots.
        cutout = self.seg_img.copy()

        # output = cv2.bitwise_and(cutout,cutout,mask=hough_bw)
        # print(cutout.shape)
        # print(hough_bw.shape)
        # cv2.imwrite("cutout.png",output)
        # mask = hough_bw

        # Extract plots
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

        for i in area_set:
            count = 0
            for area in areas:
                if area == i:
                    count += 1

            hist.append(count)

        self.plot_coords = [plot for plot in plots if plot[4] > 20000]  # ~5000 for 20mb imgs, 35000 for the 300mb ones
        # print(len(self.plot_coords))

        self.plot_coords = [plot for plot in self.plot_coords if plot[4] < 80000]
        # print(len(self.plot_coords))

        # Create images that are the final mask, as well as the original image
        # with all the non-plot regions removed.
        final_mask = np.zeros(hough_bw.shape)
        for plot in self.plot_coords:
            final_mask[plot[1]:plot[1]+plot[3],plot[0]:plot[0]+plot[2]] = 1

        plots_img = cv2.bitwise_and(cutout,cutout,mask=final_mask.astype("uint8"))

        cv2.imwrite("plots.png",plots_img)

        self.plot_coords_grids = []

        if self.plot_coords[0][0] > self.plot_coords[-1][0]:
            self.plot_coords = list(reversed(self.plot_coords))

        grid_row = 1
        grid_col = 1
        raw_y = self.plot_coords[0][1]

        for plot in self.plot_coords:
            if plot[1] != raw_y:
                grid_row += 1
                grid_col = 1
                raw_y = plot[1]
            #name = dir_name + "/" + str(grid_row) + "_" + str(grid_col) + ".png"
            #cv2.imwrite(name, plots_img[plot[1]:plot[1] + plot[3], plot[0]:plot[0] + plot[2]])
            self.plot_coords_grids.append((grid_row, grid_col, plot[0], plot[1], plot[2], plot[3], plot[4]))

            grid_col += 1


        print("Finished with segmentation, beginning data extraction")

        self.csv_path = os.path.join(self.parent_dir, "csv_data")
        self.csv_date_path = os.path.join(self.csv_path, "date_csvs")
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        if not os.path.exists(self.csv_date_path):
            os.mkdir(self.csv_date_path)

        for i in range(len(self.orthos)):
            self.get_data_series_single(i)

    # Run the pipeline for the analysis
    def run_pipeline(self, output_dir, parent_dir=None, seg_path=None, img_path=None, hmap_path=None):
        if parent_dir is None and img_path is None:
            print("Either a parent directory or image path is required, but neither was given")
            exit(1)

        if parent_dir is not None and seg_path is None:
            print("A parent directory is given, but no date for the segmentation was supplied")
            exit(2)

        if parent_dir is not None and img_path is not None:
            print("A parent directory and single image are given. Choose only one depending on whether you want to analyze a series or individual image")
            exit(3)

        self.model = load_model("model/model_5.h5")

        self.img_path = img_path
        self.hmap_path = hmap_path
        self.output_dir = output_dir
        self.parent_dir = parent_dir
        self.seg_path = seg_path

        # self.csv_path = os.path.join(self.parent_dir, "csv_data")
        # self.perform_data_transformation()
        # exit(0)

        if self.img_path is not None:
            self.single_img_analysis()
            print("single img analysis")
            # exit(0)

        if self.parent_dir is not None:
            self.series_analysis()
            print("series analysis")
            self.perform_data_transformation()
            # exit(0)
