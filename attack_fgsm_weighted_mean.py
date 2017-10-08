"""Implementation of sample attack."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cleverhans.attacks import FastGradientMethod
import numpy as np
from PIL import Image

import socket

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

import time

import inception_resnet_v2

start_time = time.time()

slim = tf.contrib.slim


model_weights={
    'ens_adv_inception_resnet_v2': 40.0,
    'ens3_adv_inception_v3': 10.0,
    'ens4_adv_inception_v3': 10.0,
    'adv_inception_v3': 5.0,
    'inception_v3': 55.0
  }


attack_banner='''

                                                                                            
                                                                                            




          _______ 
|\     /|(       )
| )   ( || () () |
| | _ | || || || |
| |( )| || |(_)| |
| || || || |   | |
| () () || )   ( |
(_______)|/     \|
                  






'''

print (attack_banner)

INCEPTION_NUM_CLASSES = 1001
INPUT_NUM_CHANNELS = 3


tf.flags.DEFINE_string(
    'local_execution', 'False', 'Use for debug only')

LOCAL_EXECUTION=(socket.gethostname()=="Titan") & (tf.flags.FLAGS.local_execution=="True")
NUM_INPUT_IMAGES_STR = "" if not LOCAL_EXECUTION else "_11"

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

IMAGES_INPUT_DIRECTORY="/media/datos/home-extension/virilo/Desktop/kaggle/nips-2017/cleverhans/examples/nips17_adversarial_competition/dataset/images"+NUM_INPUT_IMAGES_STR+"/" if LOCAL_EXECUTION else ''
    
tf.flags.DEFINE_string(
#    'input_dir', "/media/datos/home-extension/virilo/Desktop/kaggle/nips-2017/primera-ejecucion/attacks_output/fgsm/" if LOCAL_EXECUTION else '', 'Input directory with images.')
    'input_dir', IMAGES_INPUT_DIRECTORY  if LOCAL_EXECUTION else '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', "/media/datos/home-extension/virilo/Desktop/kaggle/nips-2017/output-local/10_fgsm"+NUM_INPUT_IMAGES_STR+"/" if LOCAL_EXECUTION else '', 'Output file to save labels.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS

VERBOSE_MODE=LOCAL_EXECUTION & True
LOG_ADV_DISTANCE=VERBOSE_MODE & True


print("FGSM LOCAL_EXECUTION " + NUM_INPUT_IMAGES_STR if LOCAL_EXECUTION else "FGSM")

def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


output_adv_images={}
net_names=[]
def register_images(images, filenames, net=""):
  global output_adv_images, net_names
  
  if not net in net_names:
    net_names.append(net)
  
  if not net in output_adv_images:
    output_adv_images[net]={}
  
  for i, filename in enumerate(filenames):
    output_adv_images[net][filename]=np.float32(images[i, :, :, :] )


def get_model_weight(net):
  global model_weights
  if not net in net_names:
    print("Unkown ", net)
    return 1.0
  else:
    
    return model_weights[net]

def compute_image(filename):
  global output_adv_images,net_names
  
  total_w=0.0
  for i,net in enumerate(net_names):
    w = get_model_weight(net)
    total_w+=w
    if i==0:
      img = w * output_adv_images[net][filename]
    else:
      img += w * output_adv_images[net][filename]
  
  img=img/total_w
    
  return img
  
def save_images(output_dir):
  global output_adv_images,net_names
  
  
  for i, filename in enumerate(output_adv_images[net_names[0]].keys()):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
#      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      
      img=compute_image(filename)  
      img = np.round(255.0 * (img + 1.0) * 0.5).astype(np.uint8)
      if LOG_ADV_DISTANCE:
        original_image=np.array(Image.open(os.path.join(FLAGS.input_dir, filename)).convert('RGB'))
        print(img.__class__.__name__, original_image.__class__.__name__)
        image_diff=original_image-img
        
        print("DISTANCE for ",filename, "; ", np.abs(np.max(image_diff)))
        print("STD for ",filename, "; ", np.std(image_diff))
        print("MODIFIED PIXELS ",filename, "; ", np.count_nonzero(image_diff))
      
      Image.fromarray(img).save(f, format='PNG')   

"""
def save_images(images, filenames, output_dir, prefijo=""):
  Saves images to the output directory. 

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  
  
  
  
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
#      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      img = np.round(255.0 * (np.float32(images[i, :, :, :] ) + 1.0) * 0.5).astype(np.uint8)
      if LOG_ADV_DISTANCE:
        original_image=np.array(Image.open(os.path.join(FLAGS.input_dir, filename)).convert('RGB'))
        print(img.__class__.__name__, original_image.__class__.__name__)
        image_diff=original_image-img
        
        print("DISTANCE for ",filename, "; ", np.abs(np.max(image_diff)))
        print("STD for ",filename, "; ", np.std(image_diff))
        print("MODIFIED PIXELS ",filename, "; ", np.count_nonzero(image_diff))
      
      Image.fromarray(img).save(f, format='PNG')
"""

class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs
  
class InceptionResnetV2Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
      _, end_points = inception_resnet_v2.inception_resnet_v2(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=True)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = INCEPTION_NUM_CLASSES

  tf.logging.set_verbosity(tf.logging.INFO)
  

  net='ens_adv_inception_resnet_v2'
  print(net+"\n"+str("="*len(net)))
  
  

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
      _, end_points = inception_resnet_v2.inception_resnet_v2(
          x_input, num_classes=num_classes, is_training=False)

#    predicted_labels = tf.argmax(end_points['Predictions'], 1)

    


    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionResnetV2Model(num_classes)

    fgsm = FastGradientMethod(model)
    x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)


    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=net+'.ckpt',
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        register_images(adv_images, filenames, net=net)
        
# --------------------------------------------------------------


 
  net='ens3_adv_inception_v3'
  print(net+"\n"+str("="*len(net)))

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(num_classes)

    fgsm = FastGradientMethod(model)
    x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=net+'.ckpt',
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        register_images(adv_images, filenames, net=net)
        
# --------------------------------------------------------------

  net='ens4_adv_inception_v3'
  print(net+"\n"+str("="*len(net)))

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(num_classes)

    fgsm = FastGradientMethod(model)
    x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=net+'.ckpt',
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        register_images(adv_images, filenames, net=net)
        
# --------------------------------------------------------------



  net='adv_inception_v3'
  print(net+"\n"+str("="*len(net)))

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(num_classes)

    fgsm = FastGradientMethod(model)
    x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=net+'.ckpt',
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        register_images(adv_images, filenames, net=net)
        
# --------------------------------------------------------------

  net='inception_v3'
  print(net+"\n"+str("="*len(net)))

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(num_classes)

    fgsm = FastGradientMethod(model)
    x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=net+'.ckpt',
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        register_images(adv_images, filenames, net=net)
        
# --------------------------------------------------------------








  
  save_images(FLAGS.output_dir)



if __name__ == '__main__':
  tf.app.run()

print("Executed in %s seconds." % (time.time() - start_time))
