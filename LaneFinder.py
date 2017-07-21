"""
Class for lane detection. Instance needs to be initialized with
calibration images. After calibration input images will be undistorted
before processing.

"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal


class LaneFinder:

    def __init__(self, stabilize=False):
        # flag if stabilizing should be performed
        self.stabilize = stabilize

        # parameters for camera calibration
        self.mtx = None
        self.dist = None

        # polynomial functions for left and right lanes
        self.poly_left = None
        self.poly_right = None

        # matrices for perspective transformation
        self.M = None
        self.inv_M = None

        # coordinates for perspective transform
        self.src_points = np.float32([[235, 695], [568, 467], [712, 467], [1045, 695]])
        self.target_points = np.float32([[235, 720], [235, 100], [1045, 100], [1045, 720]])

        self.kernel = None

    def calibrate_camera(self, path_test_images='camera_cal/', nx=9, ny=6, plot=False):
        """
        Calculates camera matrix and distortion factors and returns them.
        """
        camera_cal_img = [os.path.abspath(path_test_images + path) for path in os.listdir(path_test_images)]
        img_points = []  # 2d coordinates of the corners in the image
        obj_points = []  # 3d coordinates of the corners in the real thing

        objp = np.zeros([nx * ny, 3], dtype=np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        for img_path in camera_cal_img:
            img = plt.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret:
                img_points.append(corners)
                obj_points.append(objp)

                if plot:
                    plt.figure()
                    img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                    plt.imshow(img)

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    def _undistore(self, img):
        """Removes distortion due to lens. Needs calibration first."""
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def warp_image(self, img):
        """Performs perspective transform to birds eye view."""
        undist = self._undistore(img)
        if len(img.shape) == 3:
            img_size = img.shape[::-1][1:]
        elif len(img.shape) == 2:
            img_size = img.shape[::-1]

        if self.M is None:
            self.M = cv2.getPerspectiveTransform(self.src_points, self.target_points)
        warped = cv2.warpPerspective(undist, self.M, img_size, flags=cv2.INTER_LINEAR)

        return warped

    def reverse_warp_image(self, img):
        """Performs the reverse transform from birdseye to original perspective."""
        if len(img.shape) == 3:
            img_size = img.shape[::-1][1:]
        elif len(img.shape) == 2:
            img_size = img.shape[::-1]
        if self.inv_M is None:
            self.inv_M = cv2.getPerspectiveTransform(self.target_points, self.src_points)
        warped = cv2.warpPerspective(img, self.inv_M, img_size, flags=cv2.INTER_LINEAR)

        return warped

    @staticmethod
    def sharpen(img):
        """Uses a kernel to sharpen the image. Not used for processing by default."""
        kernel = np.ones([3, 3]) * -1
        kernel[1, 1] = 9
        sharp = cv2.filter2D(img, -1, kernel)
        return sharp

    @staticmethod
    def filter_gradient_magnitude(img, t_min, t_max):
        """Performs filtering via gradient magnitude and returns the filtered image."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

        mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        # scale to 8bit
        scaled = np.uint8(255 * mag / np.max(mag))

        binary_out = np.zeros_like(gray)
        binary_out[(scaled >= t_min) & (scaled <= t_max)] = 1

        return binary_out

    def _get_filtered_and_transformed(self, img, warp=True):
        """
        takes an image, applies hsv and gradient threshold and
        perspective transform.
        """

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # hsv filter yellow
        lower_filter = np.array([0, 0, 225])
        upper_filter = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_filter, upper_filter)

        # gradient magnitude filtering
        gradient_mask = self.filter_gradient_magnitude(img, 100, 255)

        combined = cv2.bitwise_or(mask, gradient_mask)

        if warp:
            combined = self.warp_image(combined)

        return combined

    @staticmethod
    def create_gaussian(window_height=80, window_width=91):
        """
        Creates and returns a 2D gaussian kernel.
        """
        x, y = np.mgrid[-window_height // 2: window_height // 2: 1,
                        -window_width // 2 + 1: window_width // 2: 1]
        # coordinates need to be stacked for distribution function
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        distribution = multivariate_normal([0, 0], cov=[[800, 0], [0, 1000]])

        return distribution.pdf(pos) * 1e4

    def _search_and_convolve(self, img, start, iteration, strides):
        """
        Takes an image, creates a subset of the image as searcharea based on the
        start position and convolves the kernel with this searcharea. Returns the
        coordinates of kernel window with highest result and its middle point's
        x_coordinate as starting point for the next iteration

        :param img: np.array img, binary img
        :param start: int, x_coordinate to define starting point
        :param iteration: int, is used to define y section of the image
        :param strides: int, kernel strides in x direction
        :return: np.array window_coordinates, int starting point for next iteration
        """
        # define searcharea
        y_abs_max = img.shape[0] - iteration * self.kernel.shape[0]
        y_abs_min = max([0, img.shape[0] - (iteration * self.kernel.shape[0] + self.kernel.shape[0])])

        x_abs_min = max([0, start - 150])
        x_abs_max = min([img.shape[1], start + 150])
        # this is the area around the starting point in which to search
        sub_img = img[y_abs_min: y_abs_max, x_abs_min: x_abs_max]

        # convolve
        # how many convolutions are necessary
        n_conv = (sub_img.shape[1] - self.kernel.shape[1]) // strides + 1

        max_conv = 0
        window_coord = None
        new_start = None
        for i in range(n_conv):
            im_ex = sub_img[:, i * strides: i * strides + self.kernel.shape[1]]

            conv = np.sum(im_ex * self.kernel)

            if conv > max_conv:
                max_conv = conv
                x_min = x_abs_min + i * strides
                x_max = x_abs_min + i * strides + self.kernel.shape[1]
                window_coord = np.array([[x_min, y_abs_min], [x_min, y_abs_max], [x_max, y_abs_min],
                                         [x_max, y_abs_max]])

                new_start = x_min + self.kernel.shape[1] // 2

        return window_coord, new_start

    def find_windows(self, img, kernel='gaussian'):
        """
        Takes in an filtered binary image and returns coordinates
        of the windows for the left and right lane.
        return array has dimensions: n_windows * 4 * 2
        """

        # create kernel/window
        if self.kernel is None:
            self.kernel = self.create_gaussian()

        n_windows = img.shape[0] // self.kernel.shape[0]

        # search for max histogram value in the lower half
        # helps with curves to search only lower half
        hist = np.sum(img[img.shape[0] // 2:], axis=0)

        # use max values in the left and right half to start window search
        start_left = np.argmax(hist[:img.shape[1] // 2])
        start_right = np.argmax(hist[img.shape[1] // 2:]) + img.shape[1] // 2

        left_lane = []
        right_lane = []

        start = start_left
        # search points for left lane
        for it in range(n_windows):
            # cut a searchable area around starting point
            window_coord, new_start = self._search_and_convolve(img, start, it, 10)

            # if no points were found, use previous starting point
            if new_start:
                start = new_start
                left_lane.append(window_coord)

        # right lane
        start = start_right

        for it in range(n_windows):
            # cut a searchable area around starting point
            window_coord, new_start = self._search_and_convolve(img, start, it, 10)

            # if no points were found, use previous starting point
            if new_start:
                start = new_start
                right_lane.append(window_coord)

        return np.stack(left_lane), np.stack(right_lane)

    @staticmethod
    def _get_points_in_window(binary_img, lane_arr):
        """
        Takes array with coordinates of all windows for a lane.
        lane_arr has dimensions n_windows * 4 * 2

        returns array with x,y coordinates for any point in
        the windows

        return dim: n_points * 2

        """
        coordinates = []
        for field in lane_arr:
            x_min = field[0, 0]
            x_max = field[2, 0]
            y_min = field[0, 1]
            y_max = field[1, 1]
            points = np.vstack(np.nonzero(binary_img[y_min:y_max, x_min:x_max])).T
            points[:, 0] += y_min
            points[:, 1] += x_min
            coordinates.append(points)

        return np.vstack(coordinates)

    @staticmethod
    def get_poly(points_arr):
        """
        Takes in an array with row, column coordinates. Fits then a second order
        polynomial and returns a function to predict x for any given y.
        """
        x = points_arr[:, 1]
        y = points_arr[:, 0]
        # fit f(y) = a*y**2 + b*y + c, x = f(y)
        poly = np.polyfit(y, x, 2)
        return np.poly1d(poly)

    def create_window_plot(self, img, plot_on_binary=False):
        """
        Takes an img and returns a color images with windows plotted on it.
        """

        # if not color image, create color dummy
        # creates the thresholded image
        binary_out = self._get_filtered_and_transformed(img)

        # use function from previous frame to make predictions
        # in order to stabilize solution and avoid jitter
        if self.stabilize:
            binary_out = self.augment_img(binary_out)

        # arrays with coordinates of the windows containing lane points
        windows_list = self.find_windows(binary_out)

        if plot_on_binary:
            img = binary_out
        else:
            img = self.warp_image(img)

        if len(img.shape) == 2 or img.shape[-1] == 1:
            img = np.dstack([img] * 3)
            overlay = np.zeros_like(img)
        elif img.shape[-1] == 3:
            overlay = np.zeros_like(img)

        for windows in windows_list:
            for window in windows:
                cv2.rectangle(overlay, tuple(window[0]), tuple(window[3]), (0, 255, 0), 7)

        return cv2.addWeighted(img, 1, overlay, 0.3, 0)

    @staticmethod
    def combine_points(x_left, y_left, x_right, y_right):
        """
        Brings arrays in the correct format for cv2.fillPoly
        """
        pts = np.int32(np.vstack([np.vstack([x_left, y_left]).T,
                                  np.flipud(np.vstack([x_right, y_right]).T)]))
        return pts

    @staticmethod
    def create_lane_plot(img, pts):
        """
        Takes an image and the estimated point coordinates and fills the lane
        with color.
        Returns the annotated image.
        """

        # if not color image, create color dummy
        if len(img.shape) == 2 or img.shape[-1] == 1:
            img = np.dstack([img] * 3)
            overlay = np.zeros_like(img)
        elif img.shape[-1] == 3:
            overlay = np.zeros_like(img)

        cv2.fillPoly(overlay, pts, [0, 255, 0])

        return cv2.addWeighted(img, 1, overlay, 0.3, 0)

    @staticmethod
    def calculate_curvature(pts):
        """
        Takes in the coordinates of the found points in pixel space
        and returns the radius of the fitted polynomial in real space

        :param pts: point coordinates in pixel space
        :return: float radius of fitted curve
        """
        # convert points to real space
        y_conv = 30 / 720
        x_conv = 3.7 / 700

        pts = np.hstack([pts[:, 0:1] * y_conv, pts[:, 1:] * x_conv])

        poly = np.poly1d(np.polyfit(pts[:, 0], pts[:, 1], 2))

        A = poly.c[0]
        B = poly.c[1]

        y_eval = 30
        curve_radius = (1 + (2 * A * y_eval + B) ** 2) ** 1.5 / np.absolute(2 * A)

        return curve_radius

    def augment_img(self, img):
        """Adds the projected polynomials to the image and returns it."""

        y = np.arange(img.shape[0])

        for predictor in [self.poly_left, self.poly_right]:
            new_pts = []
            for i in range(-1, 2):
                new_pts.append(np.int_(np.vstack([y, np.clip(predictor(y) + i, a_min=0, a_max=1279)]).T))
            pts = np.vstack(new_pts)

            img[pts[:, 0], pts[:, 1]] = (255, 0, 0)

        return img

    def _get_offset(self):
        """Calculates the offset from the middle"""

        x_left = self.poly_left(700)
        x_right = self.poly_right(700)
        x_middle = 640
        lane_width = (x_right - x_left)
        # lane in real space is 3.7 meters wide
        scale_factor = 3.7 / lane_width
        offset = (x_right - x_middle - lane_width / 2) * scale_factor

        return offset

    def mark_lane(self, img):
        """
        Takes in an image an returns an image with the lane
        highlighted in it
        """
        # creates the thresholded image
        binary_out = self._get_filtered_and_transformed(img)

        # use function from previous frame to make predictions
        # in order to stabilize solution and avoid jitter
        if self.stabilize:
            binary_out = self.augment_img(binary_out)

        # arrays with coordinates of the windows containing lane points
        windows_left_lane, windows_right_lane = self.find_windows(binary_out)

        # arrays with absolute coordinates of points within the windows
        left_pts = self._get_points_in_window(binary_out, windows_left_lane)
        right_pts = self._get_points_in_window(binary_out, windows_right_lane)

        # polynomial functions that fit each lane
        self.poly_left = self.get_poly(left_pts)
        self.poly_right = self.get_poly(right_pts)

        curve_radius = self.calculate_curvature(left_pts)
        cv2.putText(img, "Curve radius: {:5.0f} m".format(curve_radius), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1,
                   (255, 255, 255), 2, cv2.LINE_AA)
        offset = self._get_offset()
        cv2.putText(img, "Offset: {:5.2f} m".format(offset), (100, 140), cv2.FONT_HERSHEY_COMPLEX, 1,
                   (255, 255, 255), 2, cv2.LINE_AA)

        # coordinates of fitted polynomials
        y = np.arange(binary_out.shape[0])
        x_left = self.poly_left(y)
        x_right = self.poly_right(y)

        # bring coordinates in an array suitable for plotting
        lane_markings = self.combine_points(x_left, y, x_right, y)
        lane = self.create_lane_plot(binary_out, [lane_markings])

        # reverse perspective transform and overlay lane with original image
        return cv2.addWeighted(img, 1, self.reverse_warp_image(lane), 0.5, 0)

    def combine_views(self, img):
        """Creates a combined view with the lane image, window plot and transformed perspective."""
        marked_lane = self.mark_lane(img)
        window_plot = self.create_window_plot(img, plot_on_binary=True)
        window_plot = self.augment_img(window_plot)
        warped = self.warp_image(img)

        window_plot_res = cv2.resize(window_plot, (360, 360))
        warped_res = cv2.resize(warped, (360, 360))

        return np.hstack([marked_lane, np.vstack([window_plot_res, warped_res])])
