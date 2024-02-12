import keras
import numpy as np
import cv2

class Loader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, mask_img_paths,image_channel,num_classes):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.mask_img_paths = mask_img_paths
        self.num_classes = num_classes
        self.image_channel=image_channel


    def __len__(self):
        return len(self.mask_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_mask_img_paths = self.mask_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size +(self.image_channel,) , dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = cv2.resize(cv2.imread(path),self.img_size)
            x[j]=img
        y = np.zeros((self.batch_size,) + self.img_size + (self.num_classes,), dtype="uint8")
        for j, path in enumerate(batch_mask_img_paths):
            msk = cv2.resize(cv2.imread(path),self.img_size)
            y[j]=np.expand_dims((msk[:,:,0]/255).astype('uint8'),axis=-1)
        return x, y