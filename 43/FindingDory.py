import numpy as np
import cv2

from knn import KNN

class FindingDory:
    def __init__(self):
        self.light_blue = (0, 50, 100)
        self.dark_blue  = (30, 255, 255)

        self.light_yellow = (80, 140, 100)
        self.dark_yellow = (100, 255, 255)

        self.light_black = (0, 0, 0)
        self.dark_black = (255, 255, 90)

        self.knn = KNN(K=3)

    
    def convert_img_to_dataset(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        pixels_list_hsv = img_hsv.reshape(-1, 3)

        mask_blue = cv2.inRange(img_hsv, self.light_blue, self.dark_blue)
        mask_yellow = cv2.inRange(img_hsv, self.light_yellow, self.dark_yellow)
        mask_black = cv2.inRange(img_hsv, self.light_black, self.dark_black)

        mask = mask_blue + mask_yellow + mask_black

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
    finder = FindingDory()

    img = cv2.imread('Inputs/tangblue1.jpg')
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

    finder.convert_img_to_dataset(img)

    img2 = cv2.imread('Inputs/tangblue2.jpg')
    img2 = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

    mask, fish = finder.remove_background(img2)

    cv2.imwrite(fish, 'Outputs/found_dory.png')
    cv2.imwrite(mask, 'Outputs/dory_finder_mask.png')
