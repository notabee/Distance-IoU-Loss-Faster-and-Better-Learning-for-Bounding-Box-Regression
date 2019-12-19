import tensorflow as tf
import math

def compute_ciou(target,  output):
    '''
    takes in a list of bounding boxes
    but can work for a single bounding box too
    all the boundary cases such as bounding boxes of size 0 are handled.
    ''' 
    target = (target)*(target != 0)
    output = (output)*(target != 0)

    x1g, y1g, x2g, y2g = tf.split(value=target, num_or_size_splits=4, axis=1)
    x1, y1, x2, y2 = tf.split(value=output, num_or_size_splits=4, axis=1)
    
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xc1 = tf.minimum(x1, x1g)
    yc1 = tf.minimum(y1, y1g)
    xc2 = tf.maximum(x2, x2g)
    yc2 = tf.maximum(y2, y2g)
    
    ###iou term###
    xA = tf.maximum(x1g, x1)
    yA = tf.maximum(y1g, y1)
    xB = tf.minimum(x2g, x2)
    yB = tf.minimum(y2g, y2)

    interArea = tf.maximum(0.0, (xB - xA + 1)) * tf.maximum(0.0, yB - yA + 1)

    boxAArea = (x2g - x1g +1) * (y2g - y1g +1)
    boxBArea = (x2 - x1 +1) * (y2 - y1 +1)

    iouk = interArea / (boxAArea + boxBArea - interArea + 1e-10)
    ###
    
    ###distance term###
    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) +1e-7
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / c
    ###

    ###aspect-ratio term###
    arctan = tf.atan(w_gt/(h_gt + 1e-10))-tf.atan(w_pred/(h_pred + 1e-10))
    v = (4 / (math.pi ** 2)) * tf.pow((tf.atan(w_gt/(h_gt + 1e-10))-tf.atan(w_pred/(h_pred + 1e-10))),2)
    S = 1 - iouk
    alpha = v / (S + v + 1e-10)
    w_temp = 2 * w_pred
    ar = (8 / (math.pi ** 2)) * arctan * ((w_pred - w_temp) * h_pred)
    ###
    
    ###calculate diou###
    #diouk = iouk - u
    #diouk = (1 - diouk)
    ###
    
    ###calculate ciou###
    ciouk = iouk - (u + alpha * ar)
    ciouk = (1 - ciouk)
    ###
    
    return ciouk
