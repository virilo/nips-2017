#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:55:41 2017

@author: virilo
"""

import numpy as np

COLOR_DEPTH=255.0

z=[]
eps = - 2.0 * 16.0 / COLOR_DEPTH
for n in range(256):
    normalized=(np.float32(n)/ COLOR_DEPTH) * 2.0 - 1.0
    
    
    
    # we add eps
    normalized=normalized + eps
    
#    de_normalized=(((np.float32(normalized) + 1.0) * 0.5) * COLOR_DEPTH).astype(np.int16)
    de_normalized=np.round(COLOR_DEPTH * (np.float32(normalized) + 1.0) * 0.5).astype(np.int16)

    print (n, " -> ", normalized, " -> ",np.int16(de_normalized))

    z+=[de_normalized]


for i in range(256):

    if i-16 not in z:
      print("ERROR: ", i-16)

