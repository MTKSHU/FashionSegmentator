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

def change_label(im,scores,old_lb,new_lb,new_score):
    for o in old_lb:
        indexes = np.where(im == o)
        for i1,i2,i3,i4 in zip(indexes[0],indexes[1],indexes[2],indexes[3]):
            im[i1,i2,i3,i4] = new_lb
            scores[0,i2,i3] = new_score

def extract_region(im,label_used):
    a = []
    b = []
    c = []
    d = []
    for lab in label_used:
        print(lab)
        if lab != 0 and lab != 1 and lab != 2:
            indexes = np.where(im==lab)
            min_y = np.amin(indexes[0])
            max_y = np.amax(indexes[0])
            min_x = np.amin(indexes[1])
            max_x = np.amax(indexes[1])
            a.append(min_x)
            b.append(max_x)
            c.append(min_y) 
            d.append(max_y)
    bboxes = zip(a,b,c,d)
    return bboxes
    
def filter_label(im,scores,threshold):
    
    prop = regionprops(im[0,:,:,0])
    means = []
    labels_means = []
    for pr in prop:
        indexes = pr.coords
        scores_tmp = np.array(scores[0,indexes[:,0],indexes[:,1]])
        means_tmp = np.mean(scores_tmp)
        if means_tmp >= threshold :
            means.append(np.mean(scores_tmp))
            labels_means.append(im[0,indexes[0,0],indexes[0,1],0])

    print(means)
    print(labels_means)
    return means,labels_means


def predict(img_path,num_classes,model_weights,save_dir,heavy):
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
    raw_output_score = tf.reduce_max(raw_output_up,reduction_indices=[3])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    pred_score = tf.expand_dims(raw_output_score,dim=3)
  
    
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
    
        sess.run(init)
    
        # Load weights.
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess,  model_weights)
    
        # Perform inference.
        import time
        t1 = time.time()

        preds = sess.run(pred)


        # scores[0] contiene lo score massimo calcolato per ogni pixel ( 674 x 450 )
        scores = sess.run(pred_score)


    tf.reset_default_graph()
    

    msk = decode_labels(preds, num_classes= num_classes)
    """
    
    

    print('The output file has been saved to {}'.format( save_dir + 'mask.png'))
    print(time.time() - t1)"""

    means,labels_means = filter_label(msk,scores,0.98)

    my_msk = msk
    labels = []

    #Rule 1 : (skirt - shirt) vs (skirt - t-shirt) vs (dress)
    try:
        _sk = means[labels_means.index(18)]
    except ValueError:
        _sk = 0
    try:
        _sh = means[labels_means.index(15)]
    except ValueError:
        _sh = 0
    try:
        _tsh = means[labels_means.index(22)]
    except ValueError:
        _tsh = 0
    try:
        _dr = means[labels_means.index(7)]
    except ValueError:
        _dr = 0
    
    _out1 = np.mean([_sk,_sh])
    _out2 = np.mean([_sk,_tsh])
    my_max = np.argmax([_out1,_out2,_dr])
    if my_max == 0:
        change_label(my_msk,scores,[7,22],15,_sh)
    elif my_max == 1:
        change_label(my_msk,scores,[7,15],22,_tsh)
    else:
        change_label(my_msk,scores,[18,22,15],7,_dr)


    #Rule 2 : (pants) vs (leggins) vs (shorts)
    try:
        _pa = means[labels_means.index(13)]
    except ValueError:
        _pa = 0
    try:
        _leg = means[labels_means.index(21)]
    except ValueError:
        _leg = 0
    try:
        _sh = means[labels_means.index(17)]
    except ValueError:
        _sh = 0
    my_max = np.argmax([_pa,_leg,_sh])
    
    if my_max == 0:
        change_label(my_msk,scores,[21,17],13,_pa)
    elif my_max == 1:
        change_label(my_msk,scores,[13,17],21,_leg)
    else:
        change_label(my_msk,scores,[13,21],17,_sh)



    #Rule 3 : (boots) vs (shoes) vs (socks)
    try:
        _bo = means[labels_means.index(5)]
    except ValueError:
        _bo = 0
    try:
        _sh = means[labels_means.index(16)]
    except ValueError:
        _sh = 0
    try:
        _so = means[labels_means.index(19)]
    except ValueError:
        _so = 0
    my_max = np.argmax([_bo,_sh,_so])
    
    if my_max == 0:
        change_label(my_msk,scores,[16,19],5,_bo)
    elif my_max == 1:
        change_label(my_msk,scores,[5,19],16,_sh)
    else:
        change_label(my_msk,scores,[5,16],19,_so)
        
        
    #Rule 4 : (cardigan) vs (blazer)
    try:
        _ca = means[labels_means.index(20)]
    except ValueError:
        _ca = 0
    try:
        _bl = means[labels_means.index(11)]
    except ValueError:
        _bl = 0

    my_max = np.argmax([_ca,_bl])
    if heavy:
        if my_max == 0:
            change_label(my_msk,scores,[11,15,22],20,_ca)
        else:
            change_label(my_msk,scores,[20,15,22],11,_bl)
    else:
        if my_max == 0:
            change_label(my_msk,scores,[11],20,_ca)
        else:
            change_label(my_msk,scores,[20],11,_bl)
    
    #Rule 5: Coat over the all
    if heavy:
        try:
            _coat = means[labels_means.index(6)]
        except ValueError:
            _coat = 0
        if _coat != 0:
            change_label(my_msk,scores,[20,11,15,22],6,_coat)
    
    
    # Filter POST edit
    means,labels_means = filter_label(my_msk,scores,0.98)
    
    

    image = Image.fromarray(msk[0])

    if not os.path.exists( save_dir):
        os.makedirs( save_dir)
    image.save( save_dir + 'mask.png')
    


    # Create Dictionary of time
    data = {}
    data['time'] = str(time.time() -t1)
   
    # Create Dictionary of pic
    data['pic'] = img_path


    # Create Dictionary of labels used
    data['labels'] = []
    idx = 0
    for l in labels_means:
        label = {}
        label['label'] = code_colours[l]
        label['num'] = str(l)
        label['score'] = str(means[idx])
        data.get('labels').append(label)
        idx+=1

    # Create Dictionary of BoundingBox 
    bboxes = extract_region(my_msk[0],labels_means)
    data['bounding_box'] = []
    for bbox in bboxes:
        bb = {}
        bb['min_x'] = bbox[0]
        bb['max_x'] = bbox[1]
        bb['min_y'] = bbox[2]
        bb['max_y'] = bbox[3]
        data.get('bounding_box').append(bb)
    
    
    
    return image, data
    


"""if __name__ == '__main__':
    main()
"""



    