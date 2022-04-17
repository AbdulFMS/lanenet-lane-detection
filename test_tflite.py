"""
test LaneNet model on Video
"""
import argparse
import os.path as ops
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import os

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    # parser.add_argument('--weights_path', type=str, help='The model weights path')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path):
    """

    :param image_path:
    :param weights_path:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)
    
    path = '/content/res/src/'
    path1 = '/content/res/instance/'
    path2 = '/content/res/binary/'
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    isExist1 = os.path.exists(path1)
    isExist2 = os.path.exists(path2)
    
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(path)
    if not isExist1:
        # Create a new directory because it does not exist 
        os.makedirs(path1)
    if not isExist2:
        # Create a new directory because it does not exist 
        os.makedirs(path2)

    LOG.info('Start reading image and preprocessing')
    cap = cv2.VideoCapture(image_path)
    
    i = 0
    t_start = time.time()
    while cap.isOpened():
      i+=1
      try:
        # Read frame from the video
        ret, frame = cap.read()
      except:
        continue
      if ret:	
        # Detect the lanes
        read_frame(frame,i)
      else:
        break
    LOG.info('avg, cost time: {:.5f}s'.format((time.time() - t_start)/i))
    cap.release()


    return

def read_frame(frame,k):
    t_start = time.time()
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = frame
    image = image[:(2*image.shape[0])//3,...]
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    image = image.astype('float32')
    image = image[np.newaxis,...]
    # LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="models/model_float32.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    t_start = time.time()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    binary_seg_ret = interpreter.get_tensor(output_details[0]['index'])
    instance_seg_ret = interpreter.get_tensor(output_details[1]['index'])

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    t_cost = time.time() - t_start
        # t_cost /= loop_times
    # LOG.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

    postprocess_result = postprocessor.postprocess(
        binary_seg_result=binary_seg_ret[0],
        instance_seg_result=instance_seg_ret[0],
        source_image=image_vis
    )
    mask_image = postprocess_result['mask_image']
    #print(binary_seg_ret.shape)

    for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
        instance_seg_ret[0][:, :, i] = minmax_scale(instance_seg_ret[0][:, :, i])
    embedding_image = np.array(instance_seg_ret[0], np.uint8)
    #cv2.imwrite('mask_image.jpg',(mask_image * 255))
    
    bin_img = (binary_seg_ret[0] * 255).astype('uint8')
    bin_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)
    # mas = np.squeeze(mas,0).astype('uint8')
    vis_im = cv2.addWeighted(mas, 0.7, bin_img, 0.3, 0)
#     print("img",image.shape)
#     print("bin",bin_img.shape)
    cv2.imwrite(path+'src_image'+str(k)+'.jpg',vis_im)
    cv2.imwrite(path1+'instance_image'+str(k)+'.jpg',embedding_image)
    cv2.imwrite(path2+'binary_seg_image'+str(k)+'.jpg',(binary_seg_ret[0] * 255).astype('uint8'))

if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.image_path)
