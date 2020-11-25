def add_light(image, gamma=1.0,nama=name):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name + "/light-"+str(gamma)+nama+Extension, image)
    else:
        cv2.imwrite(Folder_name + "/dark-"+ str(gamma)+nama + Extension, image)
def erosion_image(image,shift,nama=name):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.erode(image,kernel,iterations = 1)
    cv2.imwrite(Folder_name + "/Erosion-"+"*"+str(shift)+nama + Extension, image)
def rotate_image(image,deg,nama=name):
    rows, cols,c = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Rotate-" + str(deg)+nama + Extension, image)
def hue_image(image,saturation,nama=name):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = image[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    image[:, :, 2] = v
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/hue-" + str(saturation)+nama + Extension, image)
def dilation_image(image,shift,nama=name):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.dilate(image,kernel,iterations = 1)
    cv2.imwrite(Folder_name + "/Dilation-" + "*" + str(shift)+nama+ Extension, image)
def median_blur(image,shift,nama=name):
    image=cv2.medianBlur(image,shift)
    cv2.imwrite(Folder_name + "/MedianBLur-" + str(shift)+nama + Extension, image)


