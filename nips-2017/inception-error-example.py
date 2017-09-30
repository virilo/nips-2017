# -*- coding: utf-8 -*-


import tensorflow as tf

# Just for debug
def debug_weights(hito="", var_name="InceptionV3/Conv2d_1a_3x3/weights:0"):
  try:
#    var=[v for v in tf.trainable_variables() if v.name == var_name][0]
    var=[v for v in tf.global_variables() if v.name == var_name][0]
    
    z=str(sess.run(var))
    print("{} --> {}...".format(hito, z[:21]))
  except Exception as ex:
    print("{} --> {}...".format(hito, type(ex).__name__))
    


from tensorflow.contrib.slim.nets import inception
import numpy as np
import os
import scipy.misc


NUM_INCEPTION_CLASSES=1001
INCEPTION_CKPT_FILE='inception_v3.ckpt'

dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint_path=dir_path + "/" + INCEPTION_CKPT_FILE 

print("checkpoint_path: " ,checkpoint_path)

slim = tf.contrib.slim

if __name__ == "__main__":
  with tf.Graph().as_default():
    
    start_vars = set(x.name for x in tf.global_variables())

    
    x_input = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(
            x_input, num_classes=NUM_INCEPTION_CLASSES , is_training=False,
            reuse=False)
    
    variables_to_restore = slim.get_model_variables()
    
    saver = tf.train.Saver(variables_to_restore)
    
    
    def predict(img):
      # Images for inception classifier are normalized to be in [-1, 1] interval.
      scaled = ((2.0 * tf.reshape(img,((-1,299,299,3))))/255) -1
      logits_tensor = tf.import_graph_def(
        sess.graph.as_graph_def(),

        input_map={'Placeholder:0': scaled},
        return_elements=['InceptionV3/Logits/SpatialSqueeze:0']) #, name="")
    
      print("logits_tensor:", logits_tensor.__class__.__name__)
      print(logits_tensor)
      
#      saver.restore(sess,checkpoint_path)
      
      return logits_tensor[0]

    with tf.Session() as sess:
      
      saver.restore(sess,checkpoint_path)
       
#      tf.logging.set_verbosity(tf.logging.INFO)
      
#     sample_image_00.png      actual label: 133
      image = np.array(scipy.misc.imresize(scipy.misc.imread('sample_image_00.png'),(299,299)),dtype=np.float32)
    
      debug_weights("just before predict(...) call")
      debug_weights("just before predict(...) call", var_name="InceptionV3/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance:0")
      model_output=predict(image).eval()
      print(model_output.shape)
      label=np.argmax(model_output)
      print("label is ", label)
      
      
      
 