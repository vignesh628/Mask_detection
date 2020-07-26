"""
@author: vignesh@rgvb
Python: 3

"""
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import time
import label_map_util, visualization_utils
from tqdm import tqdm


def create_detection_df(images_path,
                        frozen_model_path,
                        label_map_path,
                        score_thresh=0.5
                        ):
    """ Function to get detections from a frozen_model.pb

        Keyword Arguments:

        images_path --list: list of images path (No default)
        frozen_model_path --str: Path to the frozen_model.pb fil          e (No default)
        label_map_path --str: Path to label_map.pbtxt
        score_thresh --float: The minimum score to consider as a detection (Default = 0.5)

        Output
        returns a pandas Dataframe containing predictions
    """
    reverse_class_mapping_dict = label_map_util.get_label_map_dict(label_map_path=label_map_path)  # key class names
    class_mapping_dict = {v: k for k, v in reverse_class_mapping_dict.items()}  # key int
    print(images_path)
    assert isinstance(images_path, list) or isinstance(images_path, tuple), "images_path must be a list/tuple of containing full image path"
   
    df_list = []
    start_time = time.time()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.io.gfile.GFile(frozen_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print("Model loaded from disk successfully. Time taken for loading the model is {}".format(time.time()-start_time))
    time.sleep(0.2)  # for tqdm to work properly else tqdm overlaps with printing the above line
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in tqdm(images_path):
                #print(image_path)
                probs_list = []
                x1_list = []
                x2_list = []
                y1_list = []
                y2_list = []
                classes_list = []
                img_name_list = []
                #print(image_path)
                image_np = cv2.imread(image_path)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                #print(image_np)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                '''(boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np})'''
                boxes = np.squeeze(boxes)  # NOTE boxes are in normalized coordinates
                #print(boxes)
                classes = np.squeeze(classes).astype(np.int32)
                #print(classes)
                scores = np.squeeze(scores)
                #print(scores)
                #print(num)
                im_height, im_width = image_np.shape[:2]
                for box, score, clss in zip(boxes, scores, classes):
                    if score >= score_thresh:
                        box = tuple(box.tolist())
                        ymin, xmin, ymax, xmax = box
                        assert ymin < ymax and xmin < xmax
                        x1, x2, y1, y2 = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
                        x1_list.append(int(x1))
                        x2_list.append(int(x2))
                        y1_list.append(int(y1))
                        y2_list.append(int(y2))
                        probs_list.append(score*100)
                        classes_list.append(class_mapping_dict[clss])
                        #mod_image_name = image_path.split('/')[-1]
                        mod_image_name = image_path
                        img_name_list.append(mod_image_name)
                        
                df = pd.DataFrame({"image_path": img_name_list,
                                   "classes": classes_list,
                                   "score": probs_list,
                                   "x1": x1_list,
                                   "y1": y1_list,
                                   "x2": x2_list,
                                   "y2": y2_list,
                                   })
                df_list.append(df)
                #print(df_list)
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.sort_values("image_path", inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    #print(final_df)
    return(final_df)
   

def visualize_detection(dataframe,
                        label_map_path,
                        line_thickness=1
                        ):
    """ Function to visualize the detections.

    Keyword Arguments:

    dataframe --pd.Dataframe: a Pandas dataframe returned by the function create_detection_df (No default)
    label_map_path --str: Path to label_map.pbtxt
    line_thickness --int: line thickness for visualization, Default = 1

    returns:
    single image or a list of images depending on the input DataFrame

    """
    images_list = []
    reverse_class_mapping_dict = label_map_util.get_label_map_dict(label_map_path=label_map_path)  # key is class name
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=len(reverse_class_mapping_dict),
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    image_names = list(set(dataframe.image_path))
    image_names.sort()
    grouped_df = dataframe.groupby("image_path")
    for image_name in tqdm(image_names):
        df = grouped_df.get_group(image_name)
        df.reset_index(inplace=True, drop=True)
        img = cv2.imread(image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualization_utils.visualize_boxes_and_labels_on_image_array(image=img,
                                                                      boxes=np.stack((df.y1, df.x1, df.y2, df.x2), axis=1),
                                                                      classes=[reverse_class_mapping_dict[clss] for clss in df.classes],
                                                                      scores=df.score/100,
                                                                      category_index=category_index,
                                                                      instance_masks=None,
                                                                      instance_boundaries=None,
                                                                      keypoints=None,
                                                                      use_normalized_coordinates=False,
                                                                      max_boxes_to_draw=None,
                                                                      min_score_thresh=0.0,
                                                                      agnostic_mode=False,
                                                                      line_thickness=line_thickness,
                                                                      groundtruth_box_visualization_color='black',
                                                                      skip_scores=False,
                                                                      skip_labels=False
                                                                      )
        images_list.append(img)
    if len(images_list) == 1:
        return images_list[0]
    else:
        return images_list
