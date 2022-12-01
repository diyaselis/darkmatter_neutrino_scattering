from PIL import Image
import glob
import numpy as np

# Create the frames
frames = []
imgs = glob.glob("*.png")
# indx = np.random.choice(range(len(imgs)), replace=False, size=100)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('att_nevents_new.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=1000, loop=0)
