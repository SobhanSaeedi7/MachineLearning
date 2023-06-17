import os
import cv2


nums_img = cv2.imread('Inputs/numbers.jpg')

# print(nums_img.shape)

count = -1
allow = 0

for i in range(0, nums_img.shape[0], 20):
    if allow%5 == 0:
        count += 1
        path=f"Outputs/{count}"
        os.makedirs(path, exist_ok=True)
    allow += 1
    for j in range(0, nums_img.shape[1], 20):
        num = nums_img[i:i+20, j:j+20]
        cv2.imwrite(f'{path}/{count}.{j//20+1}.jpg', num)


