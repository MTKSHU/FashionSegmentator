import tensorflow as tf

def channel_intensity(image):
    im_file = tf.read_file(image)
    im = tf.image.decode_image(im_file,channels=3)
    im_float = 255*tf.image.convert_image_dtype(im,tf.float32)

    R_intensity = tf.reduce_mean(im_float[:,:,0])
    G_intensity = tf.reduce_mean(im_float[:,:,1])
    B_intensity = tf.reduce_mean(im_float[:,:,2])

    with tf.Session() as sess:
        im_file = sess.run(im_file)
        im = sess.run(im)
        im_float = sess.run(im_float)
        R_intensity = sess.run(R_intensity)
        G_intensity = sess.run(G_intensity)
        B_intensity = sess.run(B_intensity)
       
    return R_intensity, G_intensity, B_intensity