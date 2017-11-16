"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import glob
import json
from PIL import Image

import tensorflow as tf
import numpy as np
import pickle

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label, dense_crf, code_colours
from skimage import data, util
from skimage.measure import regionprops

IMG_MEAN = np.array((151.2413, 144.5654, 136.1296), dtype=np.float32)
    
NUM_CLASSES = 25
SAVE_DIR = './output/'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def predict(img_path,num_classes,model_weights,save_dir):
    """Create the model and start the evaluation process."""
    # Prepare image.
    img_orig = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img_orig)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 
    
    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes= num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])

    # CRF.
    raw_output_up = tf.nn.softmax(raw_output_up)
    raw_output_up = tf.py_func(dense_crf, [raw_output_up, tf.expand_dims(img_orig, dim=0)], tf.float32)

    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess,  model_weights)
    
    # Perform inference.
    import time
    t1 = time.time()

    preds = sess.run(pred)
    
    msk = decode_labels(preds, num_classes= num_classes)
    """
    
   

    print('The output file has been saved to {}'.format( save_dir + 'mask.png'))
    print(time.time() - t1)"""


    image = Image.fromarray(msk[0])

    if not os.path.exists( save_dir):
        os.makedirs( save_dir)
    image.save( save_dir + 'mask.png')
    

    # Create Jsonfile of time
    data = {}
    data['time'] = str(time.time() -t1)
   
   # Create Jsonfile of pic ref
    data['pic'] = img_path

    # Create JsonFile of labels used
    data['labels'] = []
    labels_used = np.unique(msk)
    for l in labels_used:
        label = {}
        label[str(l)] = code_colours[l]
        data.get('labels').append(label)

    # Create JsonFile of BoundingBox 
    props = regionprops(msk[0])
    data['bounding_box'] = []
    for pr in props:
        bb_json = {}
        bb_json['min_x'] = pr.bbox[0]
        bb_json['min_y'] = pr.bbox[1]
        bb_json['max_x'] = pr.bbox[2]
        bb_json['max_y'] = pr.bbox[3]
        data.get('bounding_box').append(bb_json)
    
    with open(save_dir+'json_data.json', 'w') as outfile:
        json.dump(data, outfile)

    json_data = json.dumps(data)
    
    return image, json_data
    


"""if __name__ == '__main__':
    main()
"""