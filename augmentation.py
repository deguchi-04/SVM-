import cv2
import numpy as np
import Augmentor

p1 = Augmentor.Pipeline("../imagesAugDouble/Aug/")
p2 = Augmentor.Pipeline("../imagesAugDouble/Aug/")
p3 = Augmentor.Pipeline("../imagesAugDouble/Aug/")
'''

p4 = Augmentor.Pipeline("../imagesAugDouble/Aug/")
p5 = Augmentor.Pipeline("../imagesAugDouble/Aug/")
p6 = Augmentor.Pipeline("../imagesAugDouble/Aug/")
'''
# Defining augmentation parameters and generating 5 samples
print("\nflip left")
p1.flip_left_right(1)
print("rotate")
p2.rotate(1, 25, 25)
print("flip bottom")
p3.flip_top_bottom(1)
'''
p3.zoom(probability = 1, min_factor = 1.1, max_factor = 1.5)
p4.shear(probability=1, max_shear_left=0.5, max_shear_right=0.5)
p5.random_distortion(probability=1, magnitude=1, grid_width=1, grid_height=1)
p6.flip_top_bottom(1)
'''


p1.sample(0)
p2.sample(0)
p3.sample(0)
'''
p3.sample(0)
p4.sample(0)
p5.sample(0)
p6.sample(0)
'''
