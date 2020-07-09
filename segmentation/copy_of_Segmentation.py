import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def downsample(image, height, inter = cv2.INTER_AREA):
    (originalHeight, originalWidth) = image.shape[:2]
    scalingFactor = height / float(originalHeight)
    dim = (int(originalWidth * scalingFactor), height)
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def saveTestImage():
    capture = cv2.VideoCapture(0)
    success = False
    while not success:
        success , image = capture.read()
    image = downsample(image, 600)
    filename = input("Enter filename: ")
    filename = filename + ".png"
    cv2.imwrite(filename, image)

def checkLiveImage():
    capture = cv2.VideoCapture(0)
    success = False
    while not success:
        success , image = capture.read()
    return downsample(image, 600)

def sortCriteriaBoundingBoxes(box):
    (x,y,w,h) = box
    return x

def findBoundingRectangle(RectangleList, imageHeight, imageWidth):
    x_min = imageWidth - 1
    y_min = imageHeight - 1
    x_max = 0
    y_max = 0
    for rect in RectangleList:
        (x, y, w, h) = rect
        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y
        if x_max  < x+w:
            x_max = x+w
        if y_max < y+h:
            y_max = y+h
    w = x_max-x_min
    h = y_max-y_min
    return (x_min, y_min, w, h)

def filterPossibleRoisByKanjiContoursColumnwise(possibleRois, grey):
    (imageHeight, imageWidth) = grey.shape[:2]
    edges = cv2.Canny(grey, 50, 100, L2gradient=True)
    rois = []
    debug = False
    for roi in possibleRois:
        (x,y,w,h) = roi
        #filter contour of whole image by assuming a contour with full height must be that one
        if h == imageHeight:
            continue
        roiImage = edges[y:y+h,x:x+w]
        contoursRoi, hierachyRoi = cv2.findContours(roiImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(x,y))
        boundingBoxes = [cv2.boundingRect(contour) for contour in contoursRoi]

        if debug:
            for box in boundingBoxes:
                (x, y, w, h) = box
                cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,0))

        #filter bounding boxes that are vertically aligned
        if len(boundingBoxes) > 0:
            boundingBoxes.sort(key=sortCriteriaBoundingBoxes)
            diff = 10
            sequences = []
            subsequence = []
            consecutive = False
            for i in range(len(boundingBoxes)-2):
                if( abs(boundingBoxes[i][0] -boundingBoxes[i+1][0]) < diff ):
                    subsequence.append(boundingBoxes[i])
                    consecutive = True
                else:
                    if consecutive:
                        subsequence.append(boundingBoxes[i])
                        consecutive = False
                        sequences.append(subsequence)
                        subsequence = []
            #Handle last element
            if consecutive:
                subsequence.append(boundingBoxes[-1])
                sequences.append(subsequence)
            else:
                last = [boundingBoxes[-1]]
                sequences.append(last)
            for subsequence in sequences:
                boundingRectangle = findBoundingRectangle(subsequence, imageHeight, imageWidth)
                rois.append(boundingRectangle)
        
    return rois

def filterPossibleRoisByKanjiContours(possibleRois, grey):
    (imageHeight, imageWidth) = grey.shape[:2]
    edges = cv2.Canny(grey, 50, 100, L2gradient=True)
    edges = cv2.dilate(edges, None)
    rois = []
    debug = False
    for roi in possibleRois:
        (x,y,w,h) = roi
        #filter contour of whole image by assuming a contour with full height must be that one
        if h == imageHeight:
            continue
        roiImage = edges[y:y+h,x:x+w]
        contoursRoi, hierachyRoi = cv2.findContours(roiImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(x,y))
        boundingBoxes = [cv2.boundingRect(contour) for contour in contoursRoi]

        if debug:
            for box in boundingBoxes:
                (x, y, w, h) = box
                cv2.rectangle(image, (x, y), (x+w, y+h), (255,255,0))

        #filter bounding boxes that are vertically aligned
        if len(boundingBoxes) > 0:
            boundingBoxes.sort(key=sortCriteriaBoundingBoxes)
            diff = 3
            sequences = []
            subsequence = []
            consecutive = False
            for i in range(len(boundingBoxes)-2):
                if( abs(boundingBoxes[i][0] -boundingBoxes[i+1][0]) < diff ):
                    subsequence.append(boundingBoxes[i])
                    consecutive = True
                else:
                    if consecutive:
                        subsequence.append(boundingBoxes[i])
                        consecutive = False
                        sequences.append(subsequence)
                        subsequence = []
            #Handle last element
            if consecutive:
                subsequence.append(boundingBoxes[-1])
                sequences.append(subsequence)
            else:
                last = [boundingBoxes[-1]]
                sequences.append(last)
            for subsequence in sequences:
                if len(subsequence)>2:
                    for rect in subsequence:
                        rois.append(rect)
        
    return rois


def filterPossibleRois(possibleRois, image, grey, imageHeight, imageWidth):
    rois = []
    for roi in possibleRois:
        (x,y,w,h) = roi
        #filter contour of whole image by assuming a contour with full height must be that one
        if h == imageHeight:
            continue
        rois.append(roi)
    return rois

def resizeRois(possibleRois, currentHeight, wantedHeight):
    scaleFactor = wantedHeight / float(currentHeight)
    scaledRois = []
    for roi in possibleRois:
        (x,y,w,h) = roi
        scaledRois.append((math.floor(x*scaleFactor), math.floor(y*scaleFactor), math.ceil(w*scaleFactor), math.ceil(h*scaleFactor)))
        
    return scaledRois
        


def segmentation(image, blackhatKernel, closingKernel):
    debug = True
    (imageHeight, imageWidth) = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    downsampledImage = downsample(grey, 480)
    (imageHeightDownsampled, imageWidthDownsampled) = downsampledImage.shape[:2]
    
    blurred = cv2.GaussianBlur(downsampledImage, (3,3), 0.5)

    adaptiveThreshold = cv2.adaptiveThreshold(src=downsampledImage, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=10)
    threshold = cv2.erode(adaptiveThreshold, None, iterations=3)#3
    #threshold = cv2.dilate(threshold,cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1) #MorphCross
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations = 4)#  blackhatKernel, iterations = 1  or cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations = 4

    harris = cv2.cornerHarris(blurred, 2,3,0.04)
    harrisRaw = np.zeros(harris.shape).astype("uint8")
    harrisRaw[harris>0.02*harris.max()]=255
    harrisRaw = cv2.morphologyEx(harrisRaw, cv2.MORPH_CLOSE, blackhatKernel, iterations = 2)
    harrisRaw = cv2.morphologyEx(harrisRaw, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), iterations=2)

    contoursHarris, hierachyHarris = cv2.findContours(harrisRaw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boundariesHarris =[]
    for contour in contoursHarris:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w*h > 10:
            boundariesHarris.append((x,y,w,h))

    contours, hierachy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    possibleRois = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        #only take those contours which encloses a group of Harris corners
        for boundary in boundariesHarris:
            (xOther, yOther, wOther, hOther) = boundary
            if x <= xOther and y <= yOther and xOther+wOther <= x+w and yOther+hOther <= y+h:
                possibleRois.append((x,y,w,h))
                break

    possibleRois = resizeRois(possibleRois, imageHeightDownsampled, imageHeight)

    rois = filterPossibleRoisByKanjiContours(possibleRois, grey)
    #rois = filterPossibleRois(possibleRois, image, grey, imageHeight, imageWidth)

    # TODO

    roi_widths = [w for (x, y, w, h) in rois]

    middle_width = 25  # np.mean(roi_widths)

    kanjis = []
    new_rois = []
    for roi in rois:
        (x, y, w, h) = roi

        if w > middle_width - 10 and w < middle_width + 10:
            v_stack_num = h / w
            sub_rois = []
            while v_stack_num > 1.75:
                sub_rois.append((x, y, w, w))
                v_stack_num = v_stack_num - 1
                y = y + w
                h = h - w
            sub_rois.append((x, y, w, h))
            for sub_roi in sub_rois:
                (x, y, w, h) = sub_roi
                kanjis.append((image[y:y+h, x:x+w], roi))
                # cv2.imshow("kanji", image[y:y+h, x:x+w])
                # cv2.waitKey(0)
            new_rois += sub_rois

    rois = new_rois

    if debug:
        for roi in rois:
            (x, y, w, h) = roi
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0))
        #cv2.imshow("Debug", image)

    print(len(possibleRois))
    print(len(rois))
    return image

def useCamera(blackhatKernel, closingKernel):
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    while True:
        success, image = capture.read()
        if not success:
            continue
        e1 = cv2.getTickCount()
        image = segmentation(image, blackhatKernel, closingKernel)
        e2 = cv2.getTickCount()
        time = (e2 - e1)/ cv2.getTickFrequency()
        print("last segmentation time:" + str(time))
        cv2.imshow("Test", image)
        if cv2.waitKey(25) == 27:
            break

def dump():
    blackhat = cv2.morphologyEx(grayImage, cv2.MORPH_BLACKHAT, blackhatKernel)
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    gradX = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, closingKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    blackhatKernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))#13,5
    blackhatKernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13))
    cannyMorph = cv2.Sobel(edges, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    cannyMorph = np.absolute(cannyMorph)
    (minVal, maxVal) = (np.min(cannyMorph), np.max(cannyMorph))
    cannyMorph = (255 * ((cannyMorph - minVal) / (maxVal - minVal))).astype("uint8")
    cannyMorph = cv2.morphologyEx(cannyMorph, cv2.MORPH_CLOSE, blackhatKernelY)
    cannyMorph = cv2.threshold(cannyMorph, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow("canny morph", cannyMorph)

    blackhatX = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, blackhatKernelX)
    blackhatY = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, blackhatKernelY)
    #cv2.imshow("Blackhat", blackhat)

    gradX = cv2.Sobel(blackhatX, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, blackhatKernelX)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow("Blackhat closed X", thresh)

    gradY = cv2.Sobel(blackhatY, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradY = np.absolute(gradY)
    (minVal, maxVal) = (np.min(gradY), np.max(gradY))
    gradY = (255 * ((gradY - minVal) / (maxVal - minVal))).astype("uint8")
    gradY = cv2.morphologyEx(gradY, cv2.MORPH_CLOSE, blackhatKernelY)
    gradY = cv2.threshold(gradY, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow("Blackhat closed Y", gradY)


def comparePipelinese(image, blackhatKernelX, blackhatKernelY, closingKernel):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Grey", grey)
    blurred = cv2.GaussianBlur(grey, (3,3), 0.5)
    mode = 2

    if mode == 1:
        #TEST BLACKHAT AND GRADIENTS
        blackhatKerneltest = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 90))
        closingKernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, blackhatKerneltest)

        gradY = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradY = np.absolute(gradY)
        (minVal, maxVal) = (np.min(gradY), np.max(gradY))
        gradY = (255 * ((gradY - minVal) / (maxVal - minVal))).astype("uint8")

        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        grad = gradX + gradY
        grad = cv2.threshold(grad, 100, 255, cv2.THRESH_BINARY)[1]
        grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, closingKernal)
        #grad = cv2.adaptiveThreshold(src=grad, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=3, C=5)

        cv2.imshow("Gradient combined", grad)
        cv2.imshow("Gradient X", gradX)
        cv2.imshow("Gradient Y", gradY)
        cv2.imshow("Blackhat", blackhat)

    if mode == 2:
        #TEST CANNY IMAGE
        edges = cv2.Canny(grey, 50, 100, L2gradient=True)
        #cv2.imshow("Canny", edges)

        #TEST ADAPTIVE THRESHOLD
        adaptiveThreshold = cv2.adaptiveThreshold(src=grey, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=11, C=10)
        #threshold = cv2.morphologyEx(adaptiveThreshold, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=4)
        #threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), iterations=1)
        threshold = cv2.erode(adaptiveThreshold, None, iterations=3)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, blackhatKernelY, iterations = 1)
        #threshold = cv2.threshold(threshold, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imshow("Adaptive Treshold", adaptiveThreshold)

        #TEST HARRIS CORNERS
        harris = cv2.cornerHarris(blurred, 2,3,0.04)
        harrisRaw = np.zeros(harris.shape)
        harrisRaw[harris>0.02*harris.max()]=255
        #result is dilated for marking the corners, not important
        harrisRaw = cv2.morphologyEx(harrisRaw, cv2.MORPH_CLOSE, blackhatKernelY, iterations = 2)
        harrisRaw = cv2.morphologyEx(harrisRaw, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), iterations=2)
        #harrisRaw = cv2.dilate(harrisRaw, None, iterations=3)
        #cv2.imshow("harris raw", harrisRaw)
        # Threshold for an optimal value, it may vary depending on the image.
        image[harrisRaw==255]=[0,0,255]
        #image[harris>0.01*harris.max()]=[0,0,255]
        #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, blackhatKernelY, iterations = 1)
        

        threshHarris = threshold.copy()
        threshHarris = harrisRaw + threshHarris
        #cv2.imshow("treshhold+Harris", threshHarris)

        # connected components, pixelwise and
        contours, hierachy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0,255,0))
        cv2.imshow("Image", image)
    
    keycode = cv2.waitKey(20000)

if __name__ == "__main__":
    blackhatKernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 3))#13,5
    blackhatKernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 7))
    closingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))#21,21
    mode = 4
    if mode == 1:
        useCamera(blackhatKernelY, closingKernel)
    if mode == 2:
        image = checkLiveImage()
        comparePipelinese(image, blackhatKernelX, blackhatKernelY, closingKernel)
    if mode == 3:
        image = cv2.imread("test.png") #remember changing directory
        comparePipelinese(image, blackhatKernelX, blackhatKernelY, closingKernel)
    if mode == 4:
        image = cv2.imread("../media/Manga_raw.jpg")
        image = segmentation(image, blackhatKernelY, closingKernel)
        cv2.imshow("Segmentation", image)
        cv2.waitKey(0)
    if mode == 5:
        saveTestImage()

    


    