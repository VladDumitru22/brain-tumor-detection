import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_random(dset1: np.ndarray, dset2: np.ndarray, num=5):
    dset1_imgs = dset1[np.random.choice(dset1.shape[0], num, replace=False)]
    dset2_imgs = dset2[np.random.choice(dset2.shape[0], num, replace=False)]

    fig, axes = plt.subplots(2, num, figsize=(16, 6))
    
    for ax, img in zip(axes[0], dset1_imgs):
        ax.imshow(np.array(img))
        ax.axis('off')
        ax.set_title("No Tumor")
        
    for ax, img in zip(axes[1], dset2_imgs):
        ax.imshow(np.array(img))
        ax.axis('off')
        ax.set_title("Tumor")
    
    plt.tight_layout()
    plt.show()





