import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# interface method for segmentation
def retrieve_current_frame(image, downsampleImageForSegmentation = False, debugLevel = 0):
    blackhatKernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 7))
    closingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    (image, grey, sequences) = segmentation(image, blackhatKernelY, closingKernel, downsampleForSegmentation=downsampleImageForSegmentation, debugLevel=debugLevel)
    rois = []
    for subsequence in sequences:
        column = []
        for box in subsequence:
            (x,y,w,h) = box
            column.append((grey[y:y+h,x:x+w], box))
        rois.append(column)
    return (image, rois)

# interface for starting the camera
def start_camera():
    global capture
    capture = cv2.VideoCapture()
    if not capture.open(0):
        print("Can not open camera")
        return
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)

# interface for getting a camera image
def get_camera_image():
    global capture
    success = False
    while not success:
        success, image = capture.read()
    return image

# interface for reading an image
def read_image(filepath: str):
    return cv2.imread(filepath)

# scale image to a specific height and preserve aspect ratio
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

# retrieve one camera picture for testing
def getLiveImage():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    success = False
    while not success:
        success , image = capture.read()
    return image

# for debugging size of Kanji rois
def histogram(rois):
    roi_widths = [w if w < 50 else 0 for (x, y, w, h) in rois]
    roi_heights = [h if h < 50 else 0 for (x, y, w, h) in rois]
    plt.hist(roi_widths)
    plt.hist(roi_heights)
    plt.show()

# function given to sort function for finding columns
def sortCriteriaPositionX(box):
    return box[0]

# function given to sort function for ordering elements within a column
def sortCriteriaPositionY(box):
    return box[1]

# find bounding rectangle for a list of bounding rectangles
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

# filter rois given by segmentation steps to have reasonable dimensions
def filterRoisBySize(possibleRois, imageHeight, imageWidth):
    rois = []
    for roi in possibleRois:
        (x,y,w,h) = roi
        if h == imageHeight:
            continue
        if w*h*5 > imageHeight*imageWidth :
            continue
        if w*h*15000< imageHeight*imageWidth or w*h <200:
            continue
        rois.append(roi)
    return rois

# resize Rois found in a downsampled image
def resizeRois(possibleRois, currentHeight, wantedHeight):
    scaleFactor = wantedHeight / float(currentHeight)
    scaledRois = []
    for roi in possibleRois:
        (x,y,w,h) = roi
        scaledRois.append((math.floor(x*scaleFactor), math.floor(y*scaleFactor), math.ceil(w*scaleFactor), math.ceil(h*scaleFactor)))
        
    return scaledRois

def postProcessColumnAndAddToList(column, columnList):
    #to filter false boxes we are not allowing single characters
    if len(column) > 1:
        #sort characters in one column
        column.sort(key=sortCriteriaPositionY)

        # remove fully contained bounding boxes
        containmentTable = np.zeros(len(column))
        for i in range(len(column)):
            (x,y,w,h) = column[i]
            j = i+1
            while j < len(column):    
                (x_n,y_n, w_n, h_n) = column[j]
                if x<=x_n and x_n+w_n<=x+w and y<=y_n and y_n+h_n<=y+h:
                    containmentTable[j]=1
                j = j+1
        column = [column[i] for i in range(len(column)) if containmentTable[i]==0]


        #check for gaps in a column
        columnPart = []
        for i in range(len(column)-1):
            (x,y,w,h) = column[i]
            (x_n,y_n, w_n, h_n) = column[i+1]
            if abs(y_n-y) < max(w,h)*1.5:
                columnPart.append(column[i])
            else:
                columnPart.append(column[i])
                columnList.append(columnPart)
                columnPart = []
        columnPart.append(column[-1])
        columnList.append(columnPart)




# find bounding boxes for Kanji in regions of first segmentation step
def filterRoisByKanjiContours(possibleRois, grey):
    debug = False
    (imageHeight, imageWidth) = grey.shape[:2]
    #use canny to find edges in image
    edges = cv2.Canny(grey, 100, 200, L2gradient=True)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT ,(3,3)))
    if debug:
        cv2.imshow("edges", edges)
    sequences = []
    
    for roi in possibleRois:
        # find contours for region of first sementation step
        (x,y,w,h) = roi
        roiImage = edges[y:y+h,x:x+w]
        contoursRoi= cv2.findContours(roiImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(x,y))[0]

        # use only those bounding boxes for reasonable contours
        boundingBoxes = [cv2.boundingRect(contour) for contour in contoursRoi]
        boundingBoxes = filterRoisBySize(boundingBoxes,imageHeight,imageWidth)

        if debug:
            imageColor = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
            for box in boundingBoxes:
                (x, y, w, h) = box
                imageColor = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(imageColor, (x, y), (x+w, y+h), (255,255,0))
            cv2.imshow("Kanji contour boxes", imageColor)
            cv2.waitKey(0)

        #construct columns of Kanji
        if len(boundingBoxes) > 0:
            #bounding boxes are vertically aligned if they are in a column
            boundingBoxes.sort(key=sortCriteriaPositionX)
            diff = 8
            subsequence = []

            for i in range(len(boundingBoxes)-1):
                (x,y,w,h) = boundingBoxes[i]
                #split boxes that are to big in roughly equally big pieces
                if h > 1.75 * w:
                    splitFactor = round(float(h)/float(w))
                    newHeight = round(h/splitFactor)
                    for j in range(splitFactor):
                        subsequence.append((x,y+j*newHeight,w,newHeight))
                else:    
                    subsequence.append(boundingBoxes[i])

                #detect column end if vertical alignment ends
                if( abs(x+w/2 -boundingBoxes[i+1][0] -boundingBoxes[i+1][2]/2) > diff ):
                    postProcessColumnAndAddToList(subsequence, sequences)
                    subsequence =[]

            #Handle last element
            subsequence.append(boundingBoxes[-1])
            postProcessColumnAndAddToList(subsequence, sequences)
            
    return sequences

def findTextFields(downsampledImage, blackhatKernel, closingKernel):
    # Kanji have a lot of Harris features so use such regions as indicators for text
    blurred = cv2.GaussianBlur(downsampledImage, (3,3), 0.5)
    harris = cv2.cornerHarris(blurred, 2,3,0.04)
    harrisRaw = np.zeros(harris.shape).astype("uint8")
    harrisRaw[harris>0.02*harris.max()]=255
    harrisRaw = cv2.morphologyEx(harrisRaw, cv2.MORPH_CLOSE, blackhatKernel, iterations = 2)
    harrisRaw = cv2.morphologyEx(harrisRaw, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), iterations=2)

    contoursHarris = cv2.findContours(harrisRaw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    boundariesHarris = []
    for contour in contoursHarris:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w*h > 10:
            boundariesHarris.append((x,y,w,h))

    # transform text fields to rectangles
    adaptiveThreshold = cv2.adaptiveThreshold(src=downsampledImage, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=9, C=10)
    threshold = cv2.erode(adaptiveThreshold, None, iterations=3)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations = 3)

    #find bounding boxes for text fields
    contours = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    possibleRois = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        #only take those contours which encloses a group of Harris corners
        for boundary in boundariesHarris:
            (xOther, yOther, wOther, hOther) = boundary
            if x <= xOther and y <= yOther and xOther+wOther <= x+w and yOther+hOther <= y+h:
                possibleRois.append((x,y,w,h))
                break
    
    return possibleRois

# main segmentation algorithm
def segmentation(image, blackhatKernel, closingKernel, downsampleForSegmentation, debugLevel = 0):
    (imageHeight, imageWidth) = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # downsample image for performance
    downsampledImage = grey
    if downsampleForSegmentation:
        downsampledImage = downsample(grey, 480)
    (imageHeightDownsampled, imageWidthDownsampled) = downsampledImage.shape[:2]

    #find regions where text could be located
    possibleRois = findTextFields(downsampledImage, blackhatKernel, closingKernel)
    if debugLevel >= 1:
        print("possibleRois: " + str(len(possibleRois)))
    
    # resize and filter rois to find Kanjis in original image
    possibleRois = filterRoisBySize(possibleRois, imageHeightDownsampled, imageWidthDownsampled)
    possibleRois = resizeRois(possibleRois, imageHeightDownsampled, imageHeight)

    #find Kanjis in text regions
    rois = filterRoisByKanjiContours(possibleRois, grey)
    
    if debugLevel>=1:
        numberRois = 0
        for column in rois:
            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            for roi in column:
                (x, y, w, h) = roi
                cv2.rectangle(image, (x, y), (x+w, y+h), color)
                numberRois = numberRois +1
        print("rois: " + str(numberRois))
    if debugLevel>=2:
        for roi in possibleRois:
            (x, y, w, h) = roi
            cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0))
    if debugLevel>=3:
        histogram(rois)  

    return (image, grey, rois)

# test segmentation pipline with camera
def useCamera(blackhatKernel, closingKernel, downsampleForSegmentation = True, debugLevel = 0):
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    while True:
        success, image = capture.read()
        if not success:
            continue
        e1 = cv2.getTickCount()
        image = segmentation(image, blackhatKernel, closingKernel, downsampleForSegmentation, debugLevel)[0]
        e2 = cv2.getTickCount()
        time = (e2 - e1)/ cv2.getTickFrequency()
        print("last segmentation time:" + str(time))
        cv2.imshow("Test", image)
        if cv2.waitKey(25) == 27:
            break


if __name__ == "__main__":
    blackhatKernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 3))#13,5
    blackhatKernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 7))
    closingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))#21,21
    mode = 2
    # create example images for testing
    if mode == 0:
        saveTestImage()
    # test camera
    if mode == 1:
        useCamera(blackhatKernelY, closingKernel, debugLevel=2)
    #test loaded images
    if mode == 2:
        image = cv2.imread("Manga_raw.jpg")
        (image, grey, rois) = segmentation(image, blackhatKernelY, closingKernel, False, debugLevel=2)
        cv2.imshow("Segmentation", image)
        cv2.waitKey(0)
        image = cv2.imread("test.png")
        (image, grey, rois) = segmentation(image, blackhatKernelY, closingKernel, True,  debugLevel=2)
        cv2.imshow("Segmentation", image)
        cv2.waitKey(0)
    # example usage for interface methods
    if mode == 3:
        start_camera()
        while True:
            image = get_camera_image()
            (image, rois) = retrieve_current_frame(image, True)
            for column  in  rois:
                for thingy in column:
                    (roi,(x,y,w,h)) = thingy
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0))
            cv2.imshow("Test", image)
            if cv2.waitKey(25) == 27:
                break 
    