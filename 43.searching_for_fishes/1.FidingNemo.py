import numpy as np
import cv2

from knn import KNN

class FindingNemo:
    def __init__(self):
        self.light_orange = (50, 190, 200)
        self.dark_orange  = (120, 255, 255)

        self.light_white = (0, 0, 200)
        self.dark_white = (145, 60, 255)

        self.light_black = (0, 0, 0)
        self.dark_black = (255, 255, 50)

        self.knn = KNN(K=3)

    
    def convert_img_to_dataset(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        pixels_list_hsv = img_hsv.reshape(-1, 3)

        mask_orange = cv2.inRange(img_hsv, self.light_orange, self.dark_orange)
        mask_white = cv2.inRange(img_hsv, self.light_white, self.dark_white)
        mask_black = cv2.inRange(img_hsv, self.light_black, self.dark_black)

        mask = mask_orange + mask_white + mask_black

        self.X_train = pixels_list_hsv / 255
        self.Y_train = mask.reshape(-1,) // 255


    def remove_background(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        self.knn.fit(self.X_train, self.Y_train)

        X_test = img_hsv.reshape(-1, 3) / 255
        Y_pred = self.knn.predict(X_test)

        output = Y_pred.reshape(img.shape[:2])

        result = cv2.bitwise_and(img, img, mask= output)


        return output, result




if __name__ == '__main__':
    finder = FindingNemo()

    img = cv2.imread('Inputs/clownfish.jpg')
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

    finder.convert_img_to_dataset(img)

    img2 = cv2.imread('Inputs/clownfish2.jpg')
    img2 = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

    mask, fish = finder.remove_background(img2)

    cv2.imwrite(fish, 'Outputs/found_nemo.png')
    cv2.imwrite(mask, 'Outputs/nemo_finder_mask.png')
