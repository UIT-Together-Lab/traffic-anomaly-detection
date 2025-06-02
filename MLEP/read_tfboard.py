import tensorflow as tf
import numpy as np
idx = 70
list_psnrs = []
list_indices = []

for e in tf.train.summary_iterator("./outputs_crossroads/summary/normal/shanghaitech/prednet_cyclegan_convlstm/events.out.tfevents.1703265817.6a048a08ce60"):
    try:
        for v in e.summary.value:
                if v.tag == 'training/positive_psnr':
                    print(idx, v.simple_value)
                    idx += 1
                    if idx % 10 == 0:
                        list_indices.append(idx)
                        list_psnrs.append(v.simple_value)
    except:
            pass
        

print(list_indices[np.argmax(list_psnrs)])
import pdb; pdb.set_trace()