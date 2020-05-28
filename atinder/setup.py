
########################################################################################

# code to move files in multiple folders into one folder # note: folder has to exist
import os
import shutil

destination = r"D:\User\Desktop\Data\Stanford Dogs\full_image_set"
path=r"D:\User\Desktop\Data\Stanford Dogs\images"
folders = os.listdir(path)

for folder in folders:
    file_path = os.path.join(path,folder)
    files = os.listdir(file_path)
    for file in files:
        source = os.path.join(file_path,file)
        dest=shutil.move(source, destination)
        

########################################################################################


# extract bounding boxes from txt files into csv (CU Dogs)
import re
import pandas as pd
import os
path = r"D:\User\Desktop\Data\Other\Heads\Columbia Dogs with Parts Dataset (Faces and face parts)\CU_Dogs\faceBoxesGT_full"
files = os.listdir(path)

df = pd.DataFrame([], columns = ['image_name' , 'class', 'xmin', 'ymin' ,
                                 'xmax', 'ymax']) 
for file in files:
    file_path = os.path.join(path,file)
    f = open(file_path, "r")
    contents = f.read()
    
    points = [int(s) for s in re.findall(r'\d+', contents)]
    xmin = min(points[0],points[6])
    ymin = min(points[1],points[3])
    xmax = max(points[2],points[4])
    ymax = max(points[5],points[7])
    
    f.close()
    
    df = df.append({'image_name': file[:-4]+'.jpg', 'class': "Dog", 'xmin': xmin,
                    'ymin': ymin, 'xmax': xmax, 'ymax': ymax},
                    ignore_index=True)

df.to_csv (r"D:\User\Desktop\Data\Other\Heads\Columbia Dogs with Parts Dataset (Faces and face parts)\CU_Dogs\annotated.csv",
           index = False, header=True)


########################################################################################


# extract bounding boxes from txt files into csv (oxford pets)
from bs4 import BeautifulSoup as BS
import re
import pandas as pd
import os
path = r"D:\User\Desktop\Data\Other\Heads\The Oxford-IIIT Pet Dataset\annotations\xmls"
files = os.listdir(path)

df = pd.DataFrame([], columns = ['image_name' , 'class', 'xmin', 'ymin' ,
                                 'xmax', 'ymax']) 
for file in files:
    file_path = os.path.join(path,file)
    f = open(file_path, "r")
    soup = BS(f)
    
    xmin = soup.findAll('xmin')[0].text
    ymin = soup.findAll('ymin')[0].text
    xmax = soup.findAll('xmax')[0].text
    ymax = soup.findAll('ymax')[0].text
    name = soup.findAll('name')[0].text.capitalize()
    f.close()
    
    df = df.append({'image_name': file[:-4]+'.jpg', 'class': name, 'xmin': xmin,
                    'ymin': ymin, 'xmax': xmax, 'ymax': ymax},
                    ignore_index=True)

df.to_csv (r"D:\User\Desktop\Data\Other\Heads\The Oxford-IIIT Pet Dataset\annotated.csv",
           index = False, header=True)

## get test iamges only

os.chdir(r'D:\User\Desktop\Data\Other\Heads\The Oxford-IIIT Pet Dataset\full_image_set')
path = r"./"
files = os.listdir(path)

df = pd.read_csv(r'../annotated.csv')

for file in files:
    if file not in df['image_name'].tolist():
        os.remove(file)
        print("File Removed!", file)
    
########################################################################################

#### convert files png to jpg
from PIL import Image
import glob
import os
os.chdir(r'D:\User\Desktop\Data\Other\Heads\makeml Pets\full_image_set')
# find png files in this directory
files = glob.glob("*.png")

for file in files:
    im = Image.open(file)
    rgb_im = im.convert('RGB')
    rgb_im.save(file.replace("png", "jpg"), quality=100)
    os.remove(file)

# extract bounding boxes from txt files into csv (makeml)
from bs4 import BeautifulSoup as BS
import re
import pandas as pd
import os
path = r"D:\User\Desktop\Data\Other\Heads\makeml Pets\annotations"
files = os.listdir(path)

df = pd.DataFrame([], columns = ['image_name' , 'class', 'xmin', 'ymin' ,
                                 'xmax', 'ymax']) 
for file in files:
    file_path = os.path.join(path,file)
    f = open(file_path, "r")
    soup = BS(f)
    
    xmin = soup.findAll('xmin')[0].text
    ymin = soup.findAll('ymin')[0].text
    xmax = soup.findAll('xmax')[0].text
    ymax = soup.findAll('ymax')[0].text
    name = soup.findAll('name')[0].text.capitalize()
    f.close()
    
    df = df.append({'image_name': file[:-4]+'.jpg', 'class': name, 'xmin': xmin,
                    'ymin': ymin, 'xmax': xmax, 'ymax': ymax},
                    ignore_index=True)

df.to_csv (r"D:\User\Desktop\Data\Other\Heads\makeml Pets\annotated.csv",
           index = False, header=True)



########################################################################################


# extract bounding boxes from txt files into csv (Stanford Dogs)
from bs4 import BeautifulSoup as BS
import re
import pandas as pd
import os
path = r"D:\User\Desktop\Data\Stanford Dogs\full_annotation"
files = os.listdir(path)

df = pd.DataFrame([], columns = ['image_name' , 'class', 'xmin', 'ymin' ,
                                 'xmax', 'ymax']) 
for file in files:
    file_path = os.path.join(path,file)
    f = open(file_path, "r")
    soup = BS(f)
    tags = soup.findAll('bndbox')
    bndboxs = []
    for tag in tags:
        bndboxs.append([int(s) for s in re.findall(r'\b\d+\b', tag.text)])
    f.close()
    
    for bndbox in bndboxs:
        df = df.append({'image_name': file+'.jpg', 'class': "Dog",
                        'xmin': bndbox[0], 'ymin': bndbox[1],
                        'xmax': bndbox[2], 'ymax': bndbox[3]},
                        ignore_index=True)

df.to_csv (r"D:\User\Desktop\Data\Stanford Dogs\annotated.csv",
           index = False, header=True)




########################################################################################




# extract bounding boxes from txt files into csv (VOC2012)
# step 1 find files with dogs in them
from bs4 import BeautifulSoup as BS
import re
import pandas as pd
import os
path = r"D:\User\Desktop\Data\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations"
files = os.listdir(path)

df = pd.DataFrame([], columns = ['image_name', 'class']) 
for file in files:
    file_path = os.path.join(path,file)
    f = open(file_path, "r")
    soup = BS(f)
    tags = soup.findAll('name')
    names = []
    for tag in tags:
        names.append(tag.text)
    f.close()
    
    for name in names:
        df = df.append({'image_name': file, 'class': name}, ignore_index=True)

# take out dog class
df = df[df['class'] == "dog"]
# want unique file names
df = df.drop_duplicates('image_name',keep='last').copy()

# move files with dogs into new folder
import os
import shutil

destination = r"D:\User\Desktop\Data\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\full_annotations"
path=r"D:\User\Desktop\Data\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations"
files = list(df['image_name'])
for file in files:
    source = os.path.join(path,file)
    dest=shutil.move(source, destination)

destination = r"D:\User\Desktop\Data\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\full_images"
path=r"D:\User\Desktop\Data\VOC2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages"
files = list(df['image_name'])
for file in files:
    file = file[0:-3]+"jpg"
    source = os.path.join(path,file)
    dest=shutil.move(source, destination)

# step 2: extract bounding boxes
from bs4 import BeautifulSoup as BS
import re
import pandas as pd
import os
path = r"D:\User\Desktop\Data\VOC2012\full_annotations"
files = os.listdir(path)

df = pd.DataFrame([], columns = ['image_name' , 'class', 'xmin', 'ymin' , 'xmax', 'ymax']) 
for file in files:
    file_path = os.path.join(path,file)
    f = open(file_path, "r")
    soup = BS(f)
    tags = soup.findAll('object')
    
    # select object classes
    anchors = soup.select('object')
    # find dogs within these
    bndboxs = []
    for anchor in anchors:
        name = anchor.find('name').text
        if name == "dog":
            bndbox=[anchor.find('xmin').text, anchor.find('ymin').text, anchor.find('xmax').text, anchor.find('ymax').text]
            bndboxs.append(bndbox)

    f.close()

    for bndbox in bndboxs:
        df = df.append({'image_name': file[0:-3]+"jpg", 'class': "Dog", 'xmin': bndbox[0], 'ymin': bndbox[1], 'xmax': bndbox[2], 'ymax': bndbox[3]}, ignore_index=True)

df.to_csv (r"D:\User\Desktop\Data\VOC2012\full_annotations\annotated.csv", index = False, header=True)

########################################################################################
# check 4 missing images
import pandas as pd
import os
file_path = r"D:\User\Desktop\Data\VOC2012\annotated.csv"
df = pd.read_csv(file_path)
df = df.drop_duplicates('image_name',keep='last').copy()
labels = list(df['image_name'])

path=r"D:\User\Desktop\Data\VOC2012\full_images"
files = os.listdir(path)

set1 = set(labels)
set2 = set(files)
missing_files = list(sorted(set1 - set2))
missing_labels = list(sorted(set2 - set1))





########################################################################################




# test plotting bounding boxes
import pandas as pd
import cv2 as cv



file_name = "n02102973_3999.jpg"
# n02107683_3828.jpg
# file_name = "2008_000078.jpg"
file_path = r"D:\User\Desktop\Data\Stanford Dogs\full_image_set"
# file_path = r"D:\User\Desktop\Data\VOC2012\full_images"
df = pd.read_csv(r"D:\User\Desktop\Data\Stanford Dogs\annotated.csv")
# df = pd.read_csv(r"D:\User\Desktop\Data\VOC2012\annotated.csv")
dogs = df[df['image_name'] == file_name]

image = cv.imread(file_path+"\\"+file_name)

for index, row in dogs.iterrows():
    xmin, ymin = row['xmin'], row['ymin']
    xmax, ymax = row['xmax'], row['ymax']
    cv.rectangle(image, (xmin,ymin), (xmax,ymax), color=(0, 255, 0), thickness=2)
    cv.namedWindow(file_name, cv.WINDOW_NORMAL)
    (h, w) = image.shape[:2]
    new_h = 1080
    new_w = int(float(w)/float(h)*1080)
    cv.resizeWindow(file_name, new_w, new_h)
    cv.imshow(file_name, image)
    cv.waitKey(0) # indefinite wait
    # cv.imwrite('messigray.png',img)
# cv.destroyAllWindows()

# old method
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
# import numpy as np
# im = np.array(Image.open(file_path+"\\"+file_name), dtype=np.uint8)
# for index, row in dogs.iterrows():
#     # Create figure and axes
#     fig,ax = plt.subplots(1)
#     # Display the image
#     ax.imshow(im)
#     # Create a Rectangle patch
#     xmin, ymin = row['xmin'], row['ymin']
#     width = row['xmax'] - row['xmin']
#     height = row['ymax'] - row['ymin']
#     rect = patches.Rectangle((xmin,ymin),width,height,linewidth=1,edgecolor='r',facecolor='none')
#     # Add the patch to the Axes
#     ax.add_patch(rect)
#     # remove axis
#     ax.set_axis_off()
#     # make background transparent
#     for item in [fig, ax]:
#         item.patch.set_visible(False) 
#     plt.show()


#########################################
    