import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt


def to_show(images, titles):
    length = len(images)
    rows = 1
    if length < 4:
        columns = length
    else:
        rows = round(length / 2)
        columns = 3
    for i in range(len(images)):
        plt.subplot(rows, columns, i + 1);
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def get_param_lines(lines):
    thetas = []
    coords = []
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        thetas.append(theta)
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        coords.append((pt1, pt2))
    return thetas, coords


def group_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_intersetion(lines1, lines2):
    pts = []
    for lineV in lines1:
        for lineH in lines2:
            xdiff = (lineV[0][0] - lineV[1][0], lineH[0][0] - lineH[1][0])
            ydiff = (lineV[0][1] - lineV[1][1], lineH[0][1] - lineH[1][1])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div == 0:
                print('lines dont have intersect')
            d = (det(*lineV), det(*lineH))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            pts.append((int(x), int(y)))
    length = len(pts)
    propably_points = np.zeros((length, 2))
    for i in range(length):
        propably_points[i] = pts[i]
    return propably_points


def transform_image(image, rect):
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))


def get_ejection_threshold(array):
    mean = np.mean(array)
    std = np.std(array)
    ejection_threshold = (mean - std, mean + std)
    return ejection_threshold, mean


def get_condition(center_point, points):
    cond_1 = 0
    cond_2 = 0
    for point in points:
        if cond_1 == 0 and center_point[1] > point[1]:
            cond_1 = 1
        elif cond_2 == 0 and center_point[1] < point[1]:
            cond_2 = 1
    return cond_1 and cond_2


def remove_res(*args):
    for arg in args:
        arg.clear()


class TakingPointsPlate:
    def __init__(self, image_path=None, stop=100, debug=0):
        self.image = cv2.imread(image_path)
        self.stop = stop
        self.debug = debug
        self.pts = []
        self.contour_points = None
        # Set vertical borders of lines
        self.shape = self.image.shape[:2][::-1]
        c = int(((self.shape[0]) / 2))
        self.thr = c if c < 100 else 100
        self.half_shape = np.array((self.shape[0] / 2, self.shape[1] / 2), dtype=np.int)
        self.processing_dict = {'image': cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), 'images_debug': [], 'titles': [],
                                'borders': [[(0, 0), (0, self.shape[1])], [(self.shape[0], 0), (self.shape[0],
                                                                                                self.shape[1])]]}

    def image_preprocessing(self):
        self.processing_dict['gray'] = cv2.cvtColor(self.processing_dict['image'], cv2.COLOR_BGR2GRAY)
        self.processing_dict['thresh'] = cv2.threshold(self.processing_dict['gray'], 0, 255,
                                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.processing_dict['edges'] = cv2.Canny(self.processing_dict['thresh'], 50, 200, apertureSize=3)
        if self.debug:
            self.processing_dict['images_debug'].extend(
                [self.processing_dict['image'], self.processing_dict['gray'], self.processing_dict['thresh'],
                 self.processing_dict['edges']])
            self.processing_dict['titles'].extend(['image', 'gray', 'thresh', 'edges'])

    def get_attitude(self, points):
        contour_image = np.array(([0, 0], [self.shape[0], 0], [self.shape[0], self.shape[1]], [0, self.shape[1]]))
        area_contour = cv2.contourArea(points)
        area_image = cv2.contourArea(contour_image)
        return True if area_contour / area_image > 0.3 else False

    def find_optimal_contour(self):
        count = 0

        while 1:
            count += 1

            if count == self.stop:
                print(f'search for lines ended at iteration {count}')
                break

            self.processing_dict['lines'] = cv2.HoughLines(self.processing_dict['edges'], rho=1, theta=np.pi / 90,
                                                           threshold=self.thr)
            if self.processing_dict['lines'] is not None:
                self.processing_dict['theta'], self.processing_dict['lines'] = get_param_lines(
                    self.processing_dict['lines'])
                ejection_threshold, mean_theta = get_ejection_threshold(self.processing_dict['theta'])

                # append best lines
                self.processing_dict['best_lines'] = []
                for t, line in zip(self.processing_dict['theta'], self.processing_dict['lines']):
                    cond = abs(t - mean_theta)
                    if ejection_threshold[0] <= t <= ejection_threshold[1] and cond < 0.1:
                        self.processing_dict['best_lines'].append(line)

                adequacy = True if len(self.processing_dict['best_lines']) >= 2 else False

                if adequacy:
                    self.pts = find_intersetion(self.processing_dict['borders'],
                                                self.processing_dict['best_lines'])

                    if len(self.pts) == 0:
                        continue
                    cond = get_condition(self.half_shape, self.pts)

                    if cond:
                        self.contour_points = group_points(self.pts)
                        attitude = self.get_attitude(self.contour_points)

                        if attitude:
                            if self.debug:
                                self.processing_dict['lines_draw'] = self.processing_dict['image'].copy()
                                for i in self.processing_dict['best_lines']:
                                    cv2.line(self.processing_dict['lines_draw'], i[0], i[1], (0, 0, 255), 1,
                                             cv2.LINE_AA)
                                self.processing_dict['images_debug'].append(self.processing_dict['lines_draw'])
                                self.processing_dict['titles'].append('lines_draw')
                            print(f'the optimality condition was satisfied on {count}')
                            break
                        else:
                            self.thr -= 1
                            remove_res(self.processing_dict["theta"],
                                       self.processing_dict['lines'],
                                       self.processing_dict['best_lines'])
                            continue
                    else:
                        self.thr -= 1
                        remove_res(self.processing_dict["theta"],
                                   self.processing_dict['lines'],
                                   self.processing_dict['best_lines'])
                        continue
                else:
                    self.thr -= 1
                    remove_res(self.processing_dict["theta"],
                               self.processing_dict['lines'],
                               self.processing_dict['best_lines'])
                    continue
            else:
                self.thr -= 1

    def alignment_plates(self):
        print(f'len order points is {len(self.pts)}')
        self.processing_dict['warped'] = transform_image(self.image, self.contour_points)
        if self.debug:
            self.processing_dict['images_debug'].append(self.processing_dict['warped'])
            self.processing_dict['titles'].append('warped')
            to_show(self.processing_dict['images_debug'], self.processing_dict['titles'])
        return self.processing_dict['warped']

    def save_results(self):
        if self.processing_dict['images_debug'] and self.processing_dict['titles'] != []:
            for image, title in zip(self.processing_dict['images_debug'], self.processing_dict['titles']):
                cv2.imwrite(f'results/{title}.png', image)
        else:
            pass


if __name__ == '__main__':
    start = time.time()
    image_path = 'test_images/num-22.png'
    get_plate = TakingPointsPlate(image_path, debug=0)
    get_plate.image_preprocessing()
    get_plate.find_optimal_contour()
    warped = get_plate.alignment_plates()
    get_plate.save_results()
    print(f'inference time is {time.time() - start}')
