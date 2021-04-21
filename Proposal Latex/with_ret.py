import cv2
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import retinex
import time
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('saya/retinex/OD/od133luxlux.mp4',fourcc,25.0,(640,480))
def load_model(model_path):
    return tf.saved_model.load(model_path)
cap = cv2.VideoCapture("/root/pengujian/video_pengujian/odsaya_133lux.mp4") 
PATH_TO_LABELS = '/root/object/models/research/object_detection/training/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
model_name = '/root/object/models/research/object_detection/inference_graph/saved_model'
detection_model = load_model(model_name)
print(detection_model.signatures['serving_default'].inputs)
detection_model.signatures['serving_default'].output_dtypes
detection_model.signatures['serving_default'].output_shapes
def highlight(image_np,value):
    cv2.putText(image_np,value ,(20, 120), cv2.FONT_HERSHEY_PLAIN, 3, (20,10,255), 2)     
def rescale_frame(frame,percent=100):
    width=int(frame.shape[1]* percent/100)
    height=int(frame.shape[0]* percent/100)
    dim = (width,height)
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
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
global mulai
mulai = time.time()   
print("waktu mulai ",mulai)
start=True
tmp=0  
counting=0 
while start:
    try:
        r, image_np = cap.read()
        ret=retinex.MSR(image_np ,[15, 80 ,250])
        output_dict = run_inference_for_single_image(detection_model, ret)
        vis_util.visualize_boxes_and_labels_on_image_array(
            ret,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            max_boxes_to_draw=1,
            min_score_thresh=.3,
            agnostic_mode=False,
            line_thickness=5)
        for index, score in enumerate(output_dict['detection_scores']):
            if score < 0.3:
                continue
            else:
                label = category_index[output_dict['detection_classes'][index]]['name']
                if(label):    
                    counting+=1
                    if(counting%20==0):
                        value=label+"detected"
                        tmp=1
        if(tmp>=1 and tmp<=5):
            highlight(ret,value)
            tmp+=1
        if tmp==5:
            tmp=counting=0
        out.write(ret)
        cv2.imshow('object_detection', image_np)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
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
    
    # out.release()
    # cap.release()
    # cv2.destroyAllWindows()

# run_inference(detection_model, cap)
# finish=time.time()-mulai

# out.release()
# cap.release()
# cv2.destroyAllWindows()
print("Done")
