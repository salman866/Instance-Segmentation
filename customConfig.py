
import os
import sys
import datetime
import numpy as np
import scipy.io
from imgaug import augmenters as iaa

# Import Mask RCNN modules
from config import Config
import utils
import model as modellib
import visualize



# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to serve as a validation set.
VAL_IMAGE_IDS = ["image_00"]


############################################################
#  Configurations
############################################################

class CustomConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (32 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101, xception
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB) [200.24, 157.31, 189.96]
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 256

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 400

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 500


class NucleusInferenceConfig(CustomConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "square"

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class NucleusDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "nucleus")
        newImageIds = []

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "test", "Train", "Test"]
        #subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        subset_dir = "train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir, 'Images')
        if subset == "val":
            newImageIds = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            #image_ids = next(os.walk(dataset_dir))[1]
            image_ids = os.listdir(dataset_dir)
            
            for image_id in image_ids:
                image_id = image_id.split('.')[0]
                newImageIds.append(image_id)

            if subset == "train":
                newImageIds = list(set(newImageIds) - set(VAL_IMAGE_IDS))

        # Add images
        for image_id in newImageIds:
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(dataset_dir, "{}.png".format(image_id)))
            

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Modied to read .mat files
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        
        mask_dir = os.path.join(os.path.split(os.path.split(info['path'])[0])[0], "Labels")

        maskFilePath = os.path.join(mask_dir,info['id']+str('.mat'))

        mask = scipy.io.loadmat(maskFilePath)['inst_map']
        
        noOfClasses = int(np.amax(mask) + 1)
        oneHotMatmask = []
        for i in range(noOfClasses):
            temp = np.zeros((mask.shape[0],mask.shape[1]))
            np.putmask(temp, mask==i, [1])
            temp = temp.astype(np.bool)
            oneHotMatmask.append(temp)
        mask = np.stack(oneHotMatmask, axis=-1)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
