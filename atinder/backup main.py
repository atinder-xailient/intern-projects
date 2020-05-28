from xailient import dnn
import cv2 as cv

file_name = "Cats_Test778"
ext = "png"

#By default Low resolution DNN for face detector will be loaded.
#To load the high resolution Face detector please comment the below lines.
#dnn.FaceDetector.set_param("resolution", dnn.RESOLUTION.HIGH)
detectum = dnn.Detector()
THRESHOLD = 0.05 # Value between 0 and 1 for confidence score

image = cv.imread('../data/'+file_name+'.'+ext)
valid, bboxes = detectum.process_frame(image, THRESHOLD)

# Loop through list (if empty this will be skipped) and overlay green bboxes
for i in bboxes:
    cv.rectangle(image, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 3)

cv.imwrite('../data/'+file_name+'_output.'+ext, image)
