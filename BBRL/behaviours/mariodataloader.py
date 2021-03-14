# Implements data loader classes for loading Mario Data. This classes overload PyTorch Dataset class.
# There 3 classes implmented, DatasetMatio is used for feature extraction while DatasetMarioBx and DatasetMarioBxv2
# are used to train behaviours
# Author: Gerardo Aragon-Camarasa, 2020

import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform, color, img_as_float, img_as_int
from PIL import Image
from scipy import misc

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, utils

class DatasetMario(Dataset):
    def __init__(self, file_path, csv_name, transform_in=None, behaviour=None):
        self.data = pd.read_csv(file_path+"/"+csv_name)
        self._path = file_path+"/"
        self.transform_in = transform_in

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load state and image from Mario FCEUX (after a dataset has been created with the Lua script and FM2)
        image_file = os.path.abspath(self.data.iloc[index, 0]) #.values.astype(np.uint8).reshape((1, 28, 28))
        state_fn = self.data.iloc[index, 1]

        # Get image
        img_rgba = Image.open(image_file)
        image_data = img_rgba.convert('RGB')

        # Get data from state.json
        with open(state_fn) as data_file:
            data = data_file.read()
            state_data = json.loads(data)

        # Get controller 1 info
        if "controller" in state_data:
            # {'A', 'B', 'down', 'left', 'right', 'select', 'start', 'up': False}
            # up, down, left, right, A, B, start, select
            control_data = np.asarray(list(state_data['controller'][0].values()), dtype=np.int)
            # control_data = np.asarray(list(range(0,len(control_data)))) * control_data
        else:
            control_data = np.zeros(8, dtype=np.int)

        # Get mario info
        if "mario" in state_data:
            mario_handler = state_data['mario']
            mpos = mario_handler["pos"]
            x1,y1,x2,y2 = mpos[0]-1,mpos[1]-8,mpos[2],mpos[3]-8 # For all items, this offsets work
            mario_handler['pos'] = [x1,y1,x2,y2]
            mario_handler['pos'] = np.asarray(mario_handler['pos'], dtype=np.int).reshape((2, 2))

            coins = mario_handler['coins']
            level_type = mario_handler['level_type']
            lives = mario_handler['lives']
            pos1, pos2 = mario_handler['pos']
            state = mario_handler['state']
            world = mario_handler['world']
            level = mario_handler['level']
            tmp_array = [state, lives, world, level, level_type, coins, pos1[0], pos1[1], pos2[0], pos2[1]]
            mario_data = np.asarray(tmp_array)
        else:
            mario_data = np.zeros(9, dtype=np.int)

        # Get enemy info
        #enemy_data = []
        enemy_data = np.asarray([0], dtype=np.int64)
        for i in range(0,5):
            key = "enemy"+str(i)
            if key in state_data:
                enemy_data = np.asarray([1], dtype=np.int64)
                #enemy_data.append(np.asarray(state_data["enemy"+str(i)], dtype=np.int))
            # else:
                # enemy_data.append(None)

        if self.transform_in is not None:
            image_data = self.transform_in(image_data)

        sample = {'image': image_data, 'state': torch.from_numpy(control_data).type(torch.long),
                  'mario': torch.from_numpy(mario_data), 'enemy': enemy_data, 'ifile': self.data.iloc[index, 0], 'sfile': self.data.iloc[index, 1]}

        return sample

class DatasetMarioBx(Dataset):

    # In this class, with a dataset (image-state pairs)
    # for each frame, we load in the image for that frame and convert it into a readable format for pytorch
    # we load in the controller data (buttons pressed) and convert it into a readable format i.e. not true/false for each button but instead numbers 

    def __init__(self, file_path, csv_name, buttontrain, transform_in=None):
        self.data = pd.read_csv(file_path+"/"+csv_name)
        self._path = file_path+"/"
        self.transform_in = transform_in
        # None is added to capture those instances where no button has been pressed
        self._buttons = {'a': 0, 'b': 1, 'down': 2, 'left': 3, 'right': 4, 'select': 5, 'start': 6, 'up': 7, 'None': 8}
        self._button = buttontrain

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load state and image from Mario FCEUX (after a dataset has been created with the Lua script and FM2)
        image_file = self._path + self.data.iloc[index, 0] #.values.astype(np.uint8).reshape((1, 28, 28))
        state_fn = self._path + self.data.iloc[index, 1]
        label_button = self.data.iloc[index, 2]

        # Get image
        img_rgba = Image.open(image_file)
        image_data = img_rgba.convert('RGB')

        # Get data from state.json
        with open(state_fn) as data_file:
            data = data_file.read()
            state_data = json.loads(data)

        # 1 means press button!
        if label_button == self._button:
            control_data = np.asarray([1], dtype=np.int64)
        else: # everything else
            control_data = np.asarray([0], dtype=np.int64)

        if self.transform_in is not None:
            image_data = self.transform_in(image_data)

        sample = {'image': image_data, 'state': torch.from_numpy(control_data).type(torch.float32), 'frame': '0'}

        return sample


class DatasetMarioBxv2(Dataset):
    # Here the loader will return i and i+1, i+2, ..., n
    # In this class, with a dataset (image-state pairs)
    # for each frame, we load in the image for that frame and convert it into a readable format for pytorch
    # we load in the controller data (buttons pressed) and convert it into a readable format i.e. not true/false for each button but instead numbers 

    def __init__(self, csv_name, buttontrain, transform_in=None):
        self.data = pd.read_csv(csv_name) # pd.read_csv(file_path+"/"+csv_name)
        self._path = "./"+"/"
        self.transform_in = transform_in
        # None is added to capture those instances where no button has been pressed
        self._buttons = {'a': 0, 'b': 1, 'down': 2, 'left': 3, 'right': 4, 'select': 5, 'start': 6, 'up': 7, 'None': 8}
        self._button = buttontrain
        self._noframes = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        images = []
        # Load state and image from Mario FCEUX (after a dataset has been created with the Lua script and FM2)
        self._noframes = self.data.iloc[index,0] # e.g for exp3 this is 3
        seq = self.data.iloc[index, 1]           # sequence number will essentially be frame number e.g. 1,2,3, n end of run. exp3 one starts at 2
        files = self.data.iloc[index, 4].replace("'","").strip('][').split(', ') # array of image files (column E in bx_data) [img1pth, img2pth, img3pth] <-python list of strings

        for ifiles in files:                    # for each image file in the array, get the image and convert it into RGB + apply the transform to it
            image_file = self._path + ifiles

            # Get image
            img_rgba = Image.open(image_file)

            image_data = img_rgba.convert('RGB')

            if self.transform_in is not None:
                image_data = self.transform_in(image_data)

            images.append(image_data)

        label_button = self.data.iloc[index, 6]  # label is from last image in the list above 
        # label_button is rbrbrb for example
        
        # CLASSIFICATION: 1 means press button!
        # if self._button in label_button:
        #     control_data = np.asarray([1], dtype=np.int64)
        # else: # everything else
        #     control_data = np.asarray([0], dtype=np.int64)

        # REGRESSION: Compute prob of pressing button in noFrames
        if type(label_button) != float:                                         # presumably an empty column will be represented as 0.0 when we load it
            prob_ocurr = label_button.count(self._button) / self._noframes      # I think this counts the number of occurences of the button we are training e.g. a, label_button = abab frames = 3, 
                                                                                # prob_occur = 2/3
            control_data = np.asarray([prob_ocurr], dtype=np.float)
        else: # everything else
            control_data = np.asarray([0.0], dtype=np.float)

        sample = {'seq': seq, 'image': images, 'state': torch.from_numpy(control_data).type(torch.float32)}   # sample = {frame_sequence_number, images_in_this sequence_converted & transformed for NN, probability of the button being trained being pressed}
        return sample


if __name__ == "__main__":
    # ***********************
    # DEMO
    # ***********************

    transform2apply = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

    dataset = DatasetMarioBxv2(csv_name=os.path.abspath("./bx_data/gerardo120719_5f.csv"), buttontrain='b', transform_in=transform2apply)
    # dataset = DatasetMarioBx(file_path="./", csv_name="./bx_data/allitems_A.csv", buttontrain='a', transform_in=transform2apply)
    # print(len(dataset))
    sample = dataset[15]
    # print(sample)

    # mario_dataset = DatasetMario(file_path="./", csv_name="allitems.csv", transform_in=transform2apply)
    # sample = mario_dataset.__getitem__(0)
    # print(sample)


class DatasetMarioFc(Dataset):

    def __init__(self, file_path, csv_name, transform_in=None):
        self.data = pd.read_csv(csv_name)
        self._path = file_path + "/"
        self.transform_in = transform_in
        self._buttons = {'a': 0, 'b': 1, 'down': 2, 'left': 3, 'right': 4, 'select': 5, 'start': 6, 'up': 7, 'None': 8}
        self._noframes = 0
        self._pressed = [0, 0, 0, 0, 0, 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        images = []
        self._noframes = self.data.iloc[index, 0]
        seq = self.data.iloc[index, 1]
        files = self.data.iloc[index, 4].replace("'", "").strip('][').split(', ')
        # array of image files (column E in bx_data) [img1pth, img2pth, img3pth] <-python list of strings

        for ifiles in files:  # for each image file in the array, get the image and convert it into RGB + apply the transform to it
            image_file = self._path + ifiles

            # Get image
            img_rgba = Image.open(image_file)

            # with Image.open(image_file) as img:
            #     width, height = img.size
            #     print(f"width: {width}\n height: {height}")
            image_data = img_rgba.convert('RGB')

            # print(image_data.shape)
            if self.transform_in is not None:
                image_data = self.transform_in(image_data)
            # print(image_data.shape)
            images.append(image_data)
        label_button = self.data.iloc[index, 6]
        if type(label_button) != float:  # presumably an empty column will be represented as 0.0 when we load it
            control_data = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float) # a, b, l, r, u, d
            control_data[0] = label_button.count('a') / self._noframes
            control_data[1] = label_button.count('b') / self._noframes
            control_data[2] = label_button.count('l') / self._noframes
            control_data[3] = label_button.count('r') / self._noframes
            control_data[4] = label_button.count('u') / self._noframes
            control_data[5] = label_button.count('d') / self._noframes
        else:  # nothing pressed
            control_data = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float)
        data = {'seq': seq, 'image': images, 'state': torch.from_numpy(control_data).type(torch.float32)}
        # print(images[0])
        return data
