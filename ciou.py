import tensorflow as tf
import math

def compute_ciou(target,  output):
    '''
    takes in a list of bounding boxes
    but can also work for a single bounding box too
    all the boundary cases such as bounding boxes of size 0 are handled.
    ''' 
    x1g, y1g, x2g, y2g = tf.split(value=target, num_or_size_splits=4, axis=1)
    x1, y1, x2, y2 = tf.split(value=output, num_or_size_splits=4, axis=1)

    x2 = tf.maximum(x1, x2)
    y2 = tf.maximum(y1, y2)
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xkis1 = tf.maximum(x1, x1g)
    ykis1 = tf.maximum(y1, y1g)
    xkis2 = tf.minimum(x2, x2g)
    ykis2 = tf.minimum(y2, y2g)

    xc1 = tf.minimum(x1, x1g)
    yc1 = tf.minimum(y1, y1g)
    xc2 = tf.maximum(x2, x2g)
    yc2 = tf.maximum(y2, y2g)

    min_ymax = tf.minimum(y2, y2g)
    max_ymin = tf.maximum(y1, y1g)
    intersect_heights = tf.maximum(0.0, min_ymax - max_ymin)
    min_xmax = tf.minimum(x2, x2g)
    max_xmin = tf.maximum(x1, x1g)
    intersect_widths = tf.maximum(0.0, min_xmax - max_xmin)

    intsctk = intersect_heights * intersect_widths
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) +1e-7
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / c

    arctan = tf.atan(w_gt/h_gt)-tf.atan(w_pred/h_pred)
    v = (4 / (math.pi ** 2)) * tf.pow((tf.atan(w_gt/h_gt)-tf.atan(w_pred/h_pred)),2)
    S = 1 - iouk
    alpha = v / (S + v)
    w_temp = 2 * w_pred
    
    ar = (8 / (math.pi ** 2)) * arctan * ((w_pred - w_temp) * h_pred)
    ciouk = iouk - (u + alpha * ar)
    ciouk = (1 - ciouk)
    ciouk = tf.where(tf.is_nan(ciouk), tf.zeros_like(ciouk), ciouk)
    return ciouk
