
# # images   

# from xailient import dnn
# import cv2 as cv
# import os
# import numpy as np

# # folder_path = '../data/'
# os.chdir(r'/media/sf_Data')
# # folder_path = './Other/Heads/makeml Pets/images/'
# folder_path = './Stanford Dogs/full_image_set/'
# SDK_name = 'Dog_Detection_Full'
# dataset_name = 'Stanford'

# sdk_folder = './Results/'+SDK_name+'/'
# save_folder = sdk_folder+dataset_name+'/'

# # get permissions for Ubuntu and create output folder
# os.umask(0)
# if not os.path.exists(sdk_folder):
#     os.mkdir(sdk_folder, mode=0o777)
# if not os.path.exists(save_folder):
#     os.mkdir(save_folder, mode=0o777)
 
# # files = os.listdir(folder_path)
# # files = ["n02085620_11238.jpg"]
# files = ["n02085620_7.jpg"]

# detectum = dnn.Detector()

# for file in files:
#     print(file, files.index(file)+1, 'of', len(files))
    
#     file_path = os.path.join(folder_path, file)
    
#     # for THRESHOLD in np.arange(0.2, 1.0, 0.1):
#     for THRESHOLD in [0.770]:  # Value between 0 and 1 for confidence score
#         image = cv.imread(file_path)
#         valid, bboxes = detectum.process_frame(image, THRESHOLD)
        
#         # remove small bounding boxes
#         removal_list = []
#         (h, w) = image.shape[:2]
#         min_size = min([h, w])
#         for i in bboxes:
#             if (i[2] - i[0] < min_size/10) & (i[3] - i[1] < min_size/10):
#                 removal_list.append(i)
#         for item in removal_list:
#             bboxes.remove(item)
            
#         # remove box inside existing box
#         removal_list = []
#         for i in bboxes:
#             for j in bboxes:
#                 if (i != j) & (i[0] >= j[0]) & (i[0] <= j[2]) & (i[2] >= j[0]) & (i[2] <= j[2]) & (i[1] >= j[1]) & (i[1] <= j[3]) & (i[3] >= j[1]) & (i[3] <= j[3]):
#                     removal_list.append(i)
#                     break
#         for item in removal_list:
#             bboxes.remove(item)
        
#         greenColor = (0, 255, 0)
#         # # Loop through list (if empty this will be skipped) and overlay green bboxes
#         # for i in bboxes:
#         #     cv.rectangle(image, (i[0], i[1]), (i[2], i[3]), color=greenColor, thickness=3)
#         #     # cv.putText(image, text="Dog", org=(i[0], i[1]), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0,0,255), thickness=1.3)
#         #     print(i)
        
#         # Add transparent box if dog present
#         if bboxes:
#             # Crop sub-rectangle from  image
#             x, y, w, h = 0, 0, w, h
#             sub_img = image[y:y+h, x:x+w]
#             green_rect = np.zeros(sub_img.shape, dtype=np.uint8)
#             green_rect[:,:] = greenColor
#             # Create new layer
#             overlay = cv.addWeighted(src1=sub_img, alpha=0.5, src2=green_rect, beta=0.5, gamma=1.0)
#             # Putting the image back to its position
#             image[y:y+h, x:x+w] = overlay
            
#         cv.imwrite(save_folder+'out_'+file[:-4]+'_'+str(round(THRESHOLD,3))+file[-4:], image)


# videos

from xailient import dnn
import cv2 as cv
import os
import numpy as np

os.chdir(r'/media/sf_Data')
folder_path = './Other/Videos/'
SDK_name = 'Dog_Detection_Full'
dataset_name = 'Video'

sdk_folder = './Results/'+SDK_name+'/'
save_folder = sdk_folder+dataset_name+'/'

# get permissions for Ubuntu and create output folder
os.umask(0)
if not os.path.exists(sdk_folder):
    os.mkdir(sdk_folder, mode=0o777)
if not os.path.exists(save_folder):
    os.mkdir(save_folder, mode=0o777)
    
# files = ["y2mate.com - Dog Walking Tips Part 1_ Techniques and Tips_qXXlICfvfbg_1080p.mp4"]
# files = ["Corgi Snow Belly Flop.mp4"]
files = ["Schedule a Dog Treat While You're Away_ Furbo Dog Camera.mp4"]

detectum = dnn.Detector()

for file in files:
    print(file)
    
    file_path = os.path.join(folder_path, file)
            
    # for THRESHOLD in np.arange(0.8, 0.95, 0.005):
    for THRESHOLD in [0.920]:  # Value between 0 and 1 for confidence score
    
        # live stream
        # file_path = "" + maybe open url
        cap = cv.VideoCapture(file_path)
        out = cv.VideoWriter(save_folder+'out_'+file[:-4]+'_'+str(round(THRESHOLD,3))+file[-4:],
                              apiPreference=int(cap.get(cv.CAP_FFMPEG)), fourcc=cv.VideoWriter_fourcc('F', 'M', 'P', '4'),
                              fps=int(cap.get(cv.CAP_PROP_FPS)),
                              frameSize=(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
        frame = 1
        while cap.grab():
            print("Frame",frame,"of",int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
            flag, image = cap.retrieve()
            if not flag: continue
            else: # There is a frame to grab
                # if (frame-1)%10 == 0: # hopefully avoids handshake failed
                #     import time
                #     time.sleep(5)
                if (frame-1)%10 == 0: # Only grab every 10th frame
                    valid, bboxes = detectum.process_frame(image, THRESHOLD)
        
                    # remove small bounding boxes
                    removal_list = []
                    (h, w) = image.shape[:2]
                    min_size = min([h, w])
                    for i in bboxes:
                        if (i[2] - i[0] < min_size/10) & (i[3] - i[1] < min_size/10):
                            removal_list.append(i)
                    for item in removal_list:
                        bboxes.remove(item)
                        
                    # remove box inside existing box
                    removal_list = []
                    for i in bboxes:
                        for j in bboxes:
                            if (i != j) & (i[0] >= j[0]) & (i[0] <= j[2]) & (i[2] >= j[0]) & (i[2] <= j[2]) & (i[1] >= j[1]) & (i[1] <= j[3]) & (i[3] >= j[1]) & (i[3] <= j[3]):
                                removal_list.append(i)
                                break
                    for item in removal_list:
                        bboxes.remove(item)
                                
                    greenColor = (0, 255, 0)
                    # # Loop through list (if empty this will be skipped) and overlay green bboxes
                    # for i in bboxes:
                    #     cv.rectangle(image, (i[0], i[1]), (i[2], i[3]), color=greenColor, thickness=3)
                    # cv.putText(image, text="Dog", org=(i[0], i[1]), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0,0,255), thickness=1.3)
                    #     print(i)
            
                    # Add transparent box if dog present
                    if bboxes:
                        # Crop sub-rectangle from  image
                        x, y, w, h = 0, 0, w, h
                        sub_img = image[y:y+h, x:x+w]
                        green_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                        green_rect[:,:] = greenColor
                        # Create new layer
                        overlay = cv.addWeighted(src1=sub_img, alpha=0.5, src2=green_rect, beta=0.5, gamma=1.0)
                        # Putting the image back to its position
                        image[y:y+h, x:x+w] = overlay    
                    
                    out.write(image)
                    
                    # cv.imshow('Video', image)
                    # if cv.waitKey(1) & 0xFF == ord('q'):
                    #     break
                else: # carry frame result over next 9 frames
                    # Add transparent box if dog present
                    if bboxes:
                        # Crop sub-rectangle from  image
                        x, y, w, h = 0, 0, w, h
                        sub_img = image[y:y+h, x:x+w]
                        green_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                        green_rect[:,:] = greenColor
                        # Create new layer
                        overlay = cv.addWeighted(src1=sub_img, alpha=0.5, src2=green_rect, beta=0.5, gamma=1.0)
                        # Putting the image back to its position
                        image[y:y+h, x:x+w] = overlay    
                    
                    out.write(image)
            frame+=1

        cap.release()
        out.release()
        cv.destroyAllWindows() 