# sudo -s for permissions

from xailient import dnn
import cv2 as cv
import os
import numpy as np
import time

# import live stream url finder
os.chdir(r'/media/sf_Data')
from Test import find_direct_url

online_link = 1 # live stream (T for next two)
video_files = 1 # videos or images
show_image = 1 # show file or save. doesn't work with images # ubuntu permissions fix: https://stackoverflow.com/questions/35486181/x-error-baddrawable-invalid-pixmap-or-window-parameter-when-launching-spyder
tinted_image = 1 # tinted image or boxes
threshold_val = [0.9] # 0.5 for whole dog, 0.65 for accurate partial dog, 0.9 for dog detection or np.arange(0.2, 1.0, 0.1) for testing
print_frame_and_time = 0

SDK_name = 'Dog_Detection_P1'
color = (0, 255, 0) # green
files = None

if not video_files:
    dataset_name = 'Stanford'
    # folder_path = './Other/Heads/makeml Pets/images/'
    folder_path = './Stanford Dogs/full_image_set/'
    # folder_path = './VOC2012/JPEGImages/'
    # files = ["n02085620_11238.jpg"]
    files = ["n02085620_3763.jpg"]
else:
    dataset_name = 'Video'
    folder_path = './Other/Videos/'
    # files = ["y2mate.com - Dog Walking Tips Part 1_ Techniques and Tips_qXXlICfvfbg_1080p.mp4"]
    files = ["Corgi Snow Belly Flop.mp4"]

if online_link: files = ["online_link.mp4"]



def create_folders(sdk_folder, save_folder):
    os.umask(0)
    if not os.path.exists(sdk_folder):
        os.mkdir(sdk_folder, mode=0o777)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder, mode=0o777)

def is_dog_present(image,bboxes):
    (h, w) = image.shape[:2]
    min_size = min([h, w])
    for i in bboxes:
        if not ((i[2] - i[0] < min_size/10) | (i[3] - i[1] < min_size/10)):
            # print("DOG FOUND!!!!")
            return True
    return False

def add_tint(image):
    # Crop sub-rectangle from  image
    (h, w) = image.shape[:2]
    x, y, w, h = 0, 0, w, h
    sub_img = image[y:y+h, x:x+w]
    green_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    green_rect[:,:] = color
    # Create new layer
    overlay = cv.addWeighted(src1=sub_img, alpha=0.5, src2=green_rect, beta=0.5, gamma=1.0)
    # Putting the image back to its position
    image[y:y+h, x:x+w] = overlay

def add_boxes(bboxes):
    for i in bboxes:
        cv.rectangle(image, (i[0], i[1]), (i[2], i[3]), color=color, thickness=3)
        # cv.putText(image, text="Dog", org=(i[0], i[1]), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0,0,255), thickness=1.3)
        # print(i)

def display_image(image):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', image)
    return cv.waitKey(1) & 0xFF == ord('q')

def process_image(image,THRESHOLD):
    start_time = time.time()
    valid, bboxes = detectum.process_frame(image, THRESHOLD)
    
    if tinted_image:
        # if not small bounding box then dog is present
        dog_present = is_dog_present(image,bboxes)
        # Add transparent box/tint if dog present
        if dog_present:
            add_tint(image)
    else:       
        # remove small bounding boxes
        removal_list = []
        (h, w) = image.shape[:2]
        min_size = min([h, w])
        for i in bboxes:
            if (i[2] - i[0] < min_size/10) | (i[3] - i[1] < min_size/10):
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
        # Loop through list (if empty this will be skipped) and overlay green bboxes
        add_boxes(bboxes)
        dog_present = False
        if bboxes:
            dog_present = True
    if print_frame_and_time: print("--- %s seconds ---" % (time.time() - start_time))
    return dog_present

sdk_folder = './Results/'+SDK_name+'/'
save_folder = sdk_folder+dataset_name+'/'
# get permissions for Ubuntu and create output folder
create_folders(sdk_folder, save_folder)
if files is None:
    files = os.listdir(folder_path)
detectum = dnn.Detector()

for file in files:
    print(file, files.index(file)+1, 'of', len(files))
    file_path = os.path.join(folder_path, file)

    for THRESHOLD in threshold_val:  # Value between 0 and 1 for confidence score
        if video_files:
            if online_link:
                cap = cv.VideoCapture()
            else:
                cap = cv.VideoCapture(file_path)
            if not show_image:
                out = cv.VideoWriter(save_folder+'out_'+file[:-4]+'_'+str(round(THRESHOLD,3))+file[-4:],
                                      apiPreference=int(cap.get(cv.CAP_FFMPEG)), fourcc=cv.VideoWriter_fourcc('F', 'M', 'P', '4'),
                                      fps=int(cap.get(cv.CAP_PROP_FPS)),
                                      frameSize=(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
            frame = 1
            
            # code to terminate loop cleanly if exit key pressed
            import threading as th
            import keyboard
            keep_going = True
            def key_capture_thread():
                global keep_going
                recorded = keyboard.record(until='esc')
                if recorded:
                    keep_going = False
            th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
            
            dog_was_present = False
            while keep_going:
                if online_link:
                    file_path = find_direct_url()
                    url_loaded = False
                    attempt = 1
                    while not url_loaded:
                        if attempt > 1: print("This is looping.. Attempt:", attempt)
                        url_loaded = cap.open(file_path) # only works with opencv 4.2 # pip install --upgrade opencv-python
                        attempt += 1

                while keep_going & cap.grab():
                    if print_frame_and_time: print("Frame",frame,"of",int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
                    frame += 1
                    flag, image = cap.retrieve()
                    if not flag: continue
                    else: # There is a frame to grab
                        dog_is_present = process_image(image,THRESHOLD)
                        
                        # prints a notification if new dog showed up
                        if (dog_is_present) & (not dog_was_present):
                            print("Dog showed up hehe")
                        dog_was_present = dog_is_present
                        
                        if show_image:
                            if display_image(image): continue   
                        else:
                            out.write(image)
            cap.release()
            cv.destroyAllWindows() 
            if not show_image: out.release()
        else:
            image = cv.imread(file_path)
            dog_is_present = process_image(image,THRESHOLD)

            if show_image:
                if display_image(image): continue 
            else:
                save_path = save_folder+'out_'+file[:-4]+'_'+str(round(THRESHOLD,3))+file[-4:]
                cv.imwrite(save_path, image)
