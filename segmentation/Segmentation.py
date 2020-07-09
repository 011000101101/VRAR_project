import cv2
import numpy as np
import math

def downsample(image, height, inter = cv2.INTER_AREA):
    (originalHeight, originalWidth) = image.shape[:2]
    scalingFactor = height / float(originalHeight)
    dim = (int(originalWidth * scalingFactor), height)
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

# wait till you get a satisfying picture. then press esc and enter filename in console
def saveTestImage():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    while True:
        success, image = capture.read()
        if not success:
            continue
        cv2.imshow("Test", image)
        if cv2.waitKey(1000) == 27:
            cv2.destroyWindow("Test")
            filename = input("Enter filename: ")
            filename = filename + ".png"
            cv2.imwrite(filename, image)
            break

def testLiveImage():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    success = False
    while not success:
        success , image = capture.read()
    return image

def sortCriteriaBoundingBoxes(box):
    return box[0]

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

def filterRoisBySize(possibleRois, imageHeight, imageWidth):
    rois = []
    for roi in possibleRois:
        (x,y,w,h) = roi
        if h == imageHeight:
            continue
        if w*h*5 > imageHeight*imageWidth:
            continue
        if w*h*15000< imageHeight*imageWidth:
            continue
        rois.append(roi)
    return rois

def filterRoisByKanjiContoursColumnwise(possibleRois, grey):
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

def filterRoisByKanjiContours(possibleRois, grey):
    (imageHeight, imageWidth) = grey.shape[:2]
    edges = cv2.Canny(grey, 100, 200, L2gradient=True)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT ,(3,3)))
    cv2.imshow("edges", edges)
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

def resizeRois(possibleRois, currentHeight, wantedHeight):
    scaleFactor = wantedHeight / float(currentHeight)
    scaledRois = []
    for roi in possibleRois:
        (x,y,w,h) = roi
        scaledRois.append((math.floor(x*scaleFactor), math.floor(y*scaleFactor), math.ceil(w*scaleFactor), math.ceil(h*scaleFactor)))
        
    return scaledRois

def segmentation(image, blackhatKernel, closingKernel, debugLevel = 0):
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
    
    if debugLevel >= 1:
        print("possibleRois: " + str(len(possibleRois)))
    
    possibleRois = filterRoisBySize(possibleRois, imageHeightDownsampled, imageWidthDownsampled)
    possibleRois = resizeRois(possibleRois, imageHeightDownsampled, imageHeight)

    #rois = filterRoisByKanjiContoursColumnwise(possibleRois, grey)
    rois = filterRoisByKanjiContours(possibleRois, grey)
    rois = filterRoisBySize(rois, imageHeight, imageWidth)

    if debugLevel>=1:
        for roi in rois:
            (x, y, w, h) = roi
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0))
        print("rois: " + str(len(rois)))
    if debugLevel>=2:
        for roi in possibleRois:
            (x, y, w, h) = roi
            cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0))

    return image

def useCamera(blackhatKernel, closingKernel, debugLevel = 0):
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    while True:
        success, image = capture.read()
        if not success:
            continue
        e1 = cv2.getTickCount()
        image = segmentation(image, blackhatKernel, closingKernel, debugLevel)
        e2 = cv2.getTickCount()
        time = (e2 - e1)/ cv2.getTickFrequency()
        print("last segmentation time:" + str(time))
        cv2.imshow("Test", image)
        if cv2.waitKey(25) == 27:
            break



if __name__ == "__main__":
    blackhatKernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 3))#13,5te
    blackhatKernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 7))
    closingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))#21,21
    mode = 1
    if mode == 0:
        saveTestImage()
    if mode == 1:
        useCamera(blackhatKernelY, closingKernel, debugLevel=1)
    if mode == 2:
        image = cv2.imread("test.png")
        image = segmentation(image, blackhatKernelY, closingKernel, debugLevel=2)
        cv2.imshow("Segmentation", image)
        cv2.waitKey(0)
    

    


    