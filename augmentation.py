import cv2
import numpy as np
import Augmentor

p = Augmentor.Pipeline("../images/")
p2 = Augmentor.Pipeline("../images/")
p3 = Augmentor.Pipeline("../images/")
p4 = Augmentor.Pipeline("../images/")
p5 = Augmentor.Pipeline("../images/")
p6 = Augmentor.Pipeline("../images/")
# Defining augmentation parameters and generating 5 samples
p.flip_left_right(1)
p2.rotate(1, 10, 10)
p3.zoom(probability = 1, min_factor = 1.1, max_factor = 1.5)
p4.shear(probability=1, max_shear_left=0.5, max_shear_right=0.5)
p5.random_distortion(probability=1, magnitude=1, grid_width=1, grid_height=1)
p6.flip_top_bottom(1)

p.sample(0)
p2.sample(0)
p3.sample(0)
p4.sample(0)
p5.sample(0)
p6.sample(0)