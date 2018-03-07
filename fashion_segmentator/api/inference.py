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
"""
code_colours = ['background', 'skin', 'hair', 'bag', 'belt',  
                'boots', 'coat', 'dress', 'glasses', 'gloves', 'hat/headband',  
                'jacket/blazer', 'necklace', 'pants/jeans', 'scarf/tie',   
                'shirt/blouse',  'shoes', 'shorts', 'skirt', 'socks', 
                'sweater/cardigan', 'tights/leggings', 'top/t-shirt',  
                'vest', 'vest2' ,'watch/bracelet']
"""
label_threshold = [0.98,0.98,0.98,0.96,0.85,
                    0.98,0.98,0.98,0.85,0.98,0.95,
                    0.98,0.90,0.98,0.85,
                    0.98,0.98,0.98,0.98,0.95,
                    0.97,0.98,0.98,
                    0.98,0.98,0.90]

MIN_INSIDE_DIM  =20


from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

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
def extract_inside_region(im,label_used,scale):
    prop = regionprops(im)    
    centroids = []
    labels = []
    for pr in prop:
        if pr.label not in [0,1,2]:
            if pr.label in label_used:
                centroids.append(pr.centroid)
                labels.append(pr.label)
    inside_box = []
    rates = []
    for c,l in zip(centroids,labels):   
        tmp = im[int(c[0])-25:int(c[0])+25,int(c[1])-25:int(c[1])+25]
        tmp_ar = np.reshape(tmp, (1,np.product(tmp.shape)))
        inside_box.append((int(((c[1])-25)*scale),int(((c[0])-25)*scale)))
        good_l = (tmp_ar == l).sum()
        rate = (good_l/2500.0)*100.0
        print(l,rate)
        rates.append(rate)
    return inside_box,rates

def filter_label(im,scores,threshold):
    
    prop = regionprops(im[0,:,:,0])
    means = []
    labels_means = []
    for pr in prop:
        indexes = pr.coords
        scores_tmp = np.array(scores[0,indexes[:,0],indexes[:,1]])
        means_tmp = np.mean(scores_tmp)
        if means_tmp >= label_threshold[pr.label] :
            means.append(np.mean(scores_tmp))
            labels_means.append(im[0,indexes[0,0],indexes[0,1],0])

    print(means)
    print(labels_means)
    return means,labels_means


def predict(preds,scores,img_path,scale,num_classes,save_dir,heavy,t1):
    
    

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
            if _ca > 0:
                change_label(my_msk,scores,[11,15,22],20,_ca)
        else:
            if _bl > 0:
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
        bb['min_x'] = int(bbox[0]*scale)
        bb['max_x'] = int(bbox[1]*scale)
        bb['min_y'] = int(bbox[2]*scale)
        bb['max_y'] = int(bbox[3]*scale)
        data.get('bounding_box').append(bb)

    bb_ins, rate = extract_inside_region(my_msk[0,:,:,0],labels_means,scale)

    data['bounding_box_inside'] = []
    for bbox in bb_ins:
        bb = {}
        bb['min_x'] = bbox[0]
        bb['max_x'] = bbox[0]+50
        bb['min_y'] = bbox[1]
        bb['max_y'] = bbox[1]+50
        data.get('bounding_box_inside').append(bb)
    return image, data
    


"""if __name__ == '__main__':
    main()
"""



    