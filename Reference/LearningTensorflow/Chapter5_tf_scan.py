import numpy as np
import tensorflow as tf

elems = np.array(["T", "e", "n", "s", "o", "r", " ", "F", "l","o","w"])
scan_sum = tf.scan(lambda a, x:a+x, elems)

sess = tf.InteractiveSession()
print(sess.run(scan_sum))

'''
[b'T' b'Te' b'Ten' b'Tens' b'Tenso' b'Tensor' b'Tensor ' b'Tensor F'
 b'Tensor Fl' b'Tensor Flo' b'Tensor Flow']
'''