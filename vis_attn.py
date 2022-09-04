from    src.utils         import  *
import numpy as np

alphas = np.random.rand(20, 12)
img = np.zeros((160, 90, 3))

visualize_attnmap_2(alphas, img)