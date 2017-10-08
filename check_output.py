from __future__ import print_function

import pandas as pd

import tensorflow as tf

import numpy as np

from PIL import Image

import os

MAX_DISTANCE=16

if not "FLAGS" in globals():
    
  tf.flags.DEFINE_string(
      'input_dir', '.../kaggle/nips-2017/cleverhans/examples/nips17_adversarial_competition/dataset/images/', 'Input directory with images.')
  
  tf.flags.DEFINE_string(
      'attacks_output_dir', '.../kaggle/nips-2017/tmp.ZZZZ/intermediate_results/attacks_output/', '...')
  
  tf.flags.DEFINE_string(
      'attack_name', 'fgsm', '...')
  
  FLAGS = tf.flags.FLAGS
  
  attack_output_dir=FLAGS.attacks_output_dir + FLAGS.attack_name

results=[]
for filepath in tf.gfile.Glob(os.path.join(attack_output_dir, '*.png')):
  with tf.gfile.Open(filepath) as f:
    
    png_filename=os.path.basename(f.name)
    
    adv_image = np.array(Image.open(f).convert('RGB')).astype(np.int16)
    original_image_filename = FLAGS.input_dir + png_filename
#    print(f.name)
#    print(original_image_filename)
    original_image=np.array(Image.open(original_image_filename).convert('RGB')).astype(np.int16)
    
    image_diff=original_image-adv_image
    
    result={
        "png_filename":png_filename,
        "DISTANCE": np.max(np.abs(image_diff)),
        "STD":np.std(image_diff),
        "MODIFIED PIXELS": np.count_nonzero(image_diff),
        "original_image values between": str( (np.min(original_image), np.max(original_image))),
        "attack output img values between": str( (np.min(adv_image), np.max(adv_image)))
    }
    
    results.append(result)

results__pd=pd.DataFrame(results)
results__pd.sort_values(by='DISTANCE', inplace=True, ascending=False)

num_errors=sum((results__pd.DISTANCE>MAX_DISTANCE).values)

if num_errors>0:

  print("ERROR!!!. Total errors: ",  num_errors)
  print("See results__pd dataframe")
else:
  print('OK!')   
