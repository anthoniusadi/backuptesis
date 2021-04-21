import cv2
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import retinex
import sys, os
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import model_from_json,load_model
import time 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dela/with retinex/ALL/ALL_dela15lux.mp4',fourcc,25.0,(640,480))
categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}
labels_dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
json_file = open("mobile_4.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("mobile_4.h5")
print("Loaded model from disk")
def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model
cap = cv2.VideoCapture("/root/pengujian/video_pengujian/ASL_Dela15lux.mp4")
PATH_TO_LABELS = '/root/object/models/research/object_detection/training/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
model_name = '/root/object/models/research/object_detection/inference_graph/saved_model'
detection_model = load_model(model_name)
print(detection_model.signatures['serving_default'].inputs)
detection_model.signatures['serving_default'].output_dtypes
detection_model.signatures['serving_default'].output_shapes
def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis]
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    return output_dict
koordinat=[]
image_width = 640
image_height =480
label=""
global mulai
mulai = time.time()  
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.shape[0],image.shape[1]
    reshaped = image.reshape((im_height, im_width, 3))
    return reshaped.astype(np.uint8)
def run_inference(model, cap):
    start= True
    while start:  
        try:
            ret, image_np = cap.read()
            ret = image_np
            ret=retinex.MSR(image_np ,[10, 80 ,250])
            output_dict = run_inference_for_single_image(model, ret)
            vis_util.visualize_boxes_and_labels_on_image_array(
                ret,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.2,
                agnostic_mode=False,
                line_thickness=5)
            for index, score in enumerate(output_dict['detection_scores']):
                if score < 0.2:
                    label=""
                    continue
                else:
                    label = category_index[output_dict['detection_classes'][index]]['name']
                    ymin, xmin, ymax, xmax = output_dict['detection_boxes'][index]
                    koordinat.append((label, int(xmin * image_width), int(ymin * image_height),int(xmax * image_width), int(ymax * image_height)))
                    ymin = (int(ymin * image_height))
                    xmin = (int(xmin * image_width))
                    xmax = (int(xmax * image_width))
                    ymax = (int(ymax * image_height))
                    roi =ret[ymin:ymax,xmin:xmax].copy()
                    new = cv2.resize(roi, (224, 224))
                    result = loaded_model.predict(new.reshape(1, 224, 224, 3)) 
                    res = 'Number Prediction : '+str(np.argmax(result))
                    cv2.putText(ret,res ,(21, 300), cv2.FONT_HERSHEY_PLAIN, 2, (20,19,255), 2)
            out.write(ret)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        except:
            selesai=time.time()
            print("waktu selesai",selesai)
            print("done")
            waktukomp= selesai - mulai
            print(" waktu komputasi ",waktukomp)
            out.release()
            cap.release()
            cv2.destroyAllWindows()
            start=False
run_inference(detection_model, cap)
cap.release()
cv2.destroyAllWindows()
