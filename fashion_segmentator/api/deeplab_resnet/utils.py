from PIL import Image
import numpy as np
import tensorflow as tf
import pydensecrf.densecrf as dcrf
import random

# # colour map
# label_colours = [(0,0,0)
#                 # 0=background
#                 ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
#                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
#                 ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
#                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
#                 ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
#                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
#                 ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
#                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
#                 ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
#                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
#                 ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
#                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
#                 ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
#                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
#                 ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
#                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
#                 ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
#                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
#                 ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128),(192,128,128)
#                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
#                 ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
#                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


# label_colours = [(0, 0, 0),#background
# (226, 196, 196),#skin
# (64, 32, 32),#hair
# (255, 0, 0),#bag
# (255, 191, 0),#belt
# (128, 255, 0),#boots
# (0, 255, 64),#coat
# (0, 255, 255),#dress
# (0, 64, 255),#glasses
# (128, 0, 255),#gloves
# (255, 0, 191),#hat/headband
# (255, 85, 85),#jacket/blazer
# (255, 231, 85),#necklace
# (134, 255, 85),#pants/jeans
# (85, 255, 182),#scarf/tie
# (85, 182, 255),#shirt/blouse
# (134, 85, 255),#shoes
# (255, 85, 231),#shorts
# (255, 170, 170),#skirt
# (255, 243, 170),#socks
# (194, 255, 170),#sweater/cardigan
# (170, 255, 219),#tights/leggings
# (170, 219, 255),#top/t-shirt
# (194, 170, 255),#vest
# (100, 200, 100),#vest
# (255, 170, 243)]#watch/bracelet

code_colours = ['background', 'skin', 'hair', 'bag', 'belt',  
                'boots', 'coat', 'dress', 'glasses', 'gloves', 'hat/headband',  
                'jacket/blazer', 'necklace', 'pants/jeans', 'scarf/tie',   
                'shirt/blouse',  'shoes', 'shorts', 'skirt', 'socks', 
                'sweater/cardigan', 'tights/leggings', 'top/t-shirt',  
                'vest', 'vest2' ,'watch/bracelet'] 

# label_colours = [(128,0,0), (0,0,0), (255,0,0), (255,99,71), (255,127,80),(205,92,92),
#                (240,128,128),(233,150,122),(250,128,114),(255,160,122),(255,69,0),(255,140,0),
#                (255,165,0),(255,215,0),(184,134,11),(218,165,32),(238,232,170),(189,183,107),
#                (240,230,140),(128,128,0),(138,43,226),(255,255,0),(154,205,50),(85,107,47),(107,142,35),
#                (173,255,47),(0,100,0),(0,128,0),(0,255,0),(50,205,50),(143,188,143),(46,139,87),
#                (102,205,170),(60,179,113),(32,178,170),(47,79,79),(0,139,139),(0,255,255),
#                (224,255,255),(0,206,209),(64,224,208),(175,238,238),(95,158,160),(70,130,180),
#                (100,149,237),(0,191,255),(30,144,255),(173,216,230),(135,206,235),(0,0,128),
#                (0,0,205),(0,0,255),(65,105,225),(75,0,130),(72,61,139),(106,90,205), (50, 100, 150)]  
#random.shuffle(label_colours)
#label_colours[1] = (0, 0, 0)


label_colours = [(x, x, x) for x in range(56)]

IMG_MEAN_DEFAULT = np.array((151.2412, 144.5654, 136.1296), dtype=np.float32)
n_classes = 25


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    data = tf.import_graph_def(graph_def, name="", return_elements=['DecodeJpeg:0','fc1_voc12:0'],)
    inputs = data[0]
    raw_output = data[1]
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(inputs)[0:2,])
    # CRF.
    raw_output_up = tf.nn.softmax(raw_output_up)
    raw_output_up = tf.py_func(dense_crf, [raw_output_up, tf.expand_dims(inputs, dim=0)], tf.float32)
    raw_output_score = tf.reduce_max(raw_output_up,reduction_indices=[3])
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    pred_score = tf.expand_dims(raw_output_score,dim=3)    
    return inputs, pred, pred_score

def decode_labels(mask, num_images=1, num_classes=n_classes):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
      num_classes: number of classes to predict (including background).
      one_hot: whether perform one-hot encoding.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch

def inv_preprocess(imgs, num_images=1, img_mean=IMG_MEAN_DEFAULT):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + img_mean)[:, :, ::-1].astype(np.uint8)
    return outputs

def dense_crf(probs, img=None, n_iters=10, 
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
    
    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      
    Returns:
      Refined predictions after MAP inference.
    """
    _, h, w, _ = probs.shape
    
    probs = probs[0].transpose(2, 0, 1).copy(order='C') # Need a contiguous array.
    
    d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
    U = -np.log(probs) # Unary potential.
    U = U.reshape((n_classes, -1)) # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert(img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)
