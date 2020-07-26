#!/usr/bin/env python
# coding: utf-8
import os
import subprocess
import PyPDF2
import glob
import time
import cv2
import pandas as pd
from PIL import Image, ImageSequence
import shutil
import tensorflow
from infer_detections import create_detection_df
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from bs4 import BeautifulSoup
# In[47]:
def create_dirs(out_jpg_dir,out_xml_dir,form_type_path):
    if os.path.exists('home/developer/Extraction/DATA/OUTPUT/'):
        shutil.rmtree('/home/developer/Extraction/DATA/OUTPUT/')
    if not os.path.exists(out_jpg_dir):
        os.makedirs(out_jpg_dir)
    if not os.path.exists(out_xml_dir):
        os.makedirs(out_xml_dir)
    if not os.path.exists(form_type_path[0]):
        os.makedirs(form_type_path[0])
    if not os.path.exists(form_type_path[1]):
        os.makedirs(form_type_path[1])
#    if not os.path.exists(form_type_path[2]):
#        os.makedirs(form_type_path[2])


# In[48]:

def tiff_to_jpg(tif_path,out_jpg_dir):
    img_list = glob.glob(tif_path)
    for j in img_list:
        img = cv2.imreadmulti(j)
        input_img = os.path.basename(j).split('.')[0].split('_')[0]
        im_path = os.path.join(out_jpg_dir,input_img)
        if not os.path.exists(im_path):
            os.mkdir(im_path)
        for i in range(len(img[1])):
            upd_i = i+1
            im_name = input_img+"_"+str(upd_i)+".jpg"
            im_name = os.path.join(im_path,im_name)
            try:
                cv2.imwrite(im_name, img[1][i])
            except Exception as e:
                print(e)
# In[76]:

def pdf_convert_text(pdf_path, save_path, page_num):
    if os.path.exists(save_path) == False:
        os.chdir(r"C:\Users\a\Downloads\Extraction\poppler-0.68.0\bin")
        argument = 'pdftotext.exe -bbox-layout' + ' -f ' + str(page_num) + ' -l ' + str(page_num) + ' ' + pdf_path + ' ' + save_path
        #argument = argument.split()
        mysubproces = subprocess.Popen(argument)
        mysubproces.wait()



def extract_text(pdf_path,out_xml_dir):
    pdf_object = PyPDF2.PdfFileReader(pdf_path)
    page_number = pdf_object.getNumPages()
    ind_xml_path = out_xml_dir+os.path.basename(pdf_path).split('.')[0]
    if not os.path.exists(ind_xml_path):
        os.makedirs(ind_xml_path)
    for page_num in range(1,page_number+1):
        save_path=ind_xml_path+'\\'+os.path.basename(pdf_path).split('.')[0]+'_'+str(page_num)+'.xml'
        pdf_convert_text(pdf_path,save_path,page_num)



def xml_convertion(pdf_path,out_xml_dir):
    pdf_list=os.listdir(pdf_path)
    for i in tqdm(pdf_list):
        path = os.path.join(pdf_path,i)
        extract_text(path,out_xml_dir)


# In[77]:



def str_replace_xml(xml_str):
    xml_str=xml_str.replace('http://www.w3.org/1999/xhtml','')
    xml_str=xml_str.replace('❑','')
    xml_str=xml_str.replace('•','')
    xml_str=xml_str.replace('■','')
    return(xml_str)


def move_folder(out_jpg_dir,out_xml_dir,form_type_path,folder):
    form_jpg_path = os.path.join(form_type_path,folder,'jpg')
    form_xml_path = os.path.join(form_type_path,folder,'xml')
    shutil.move(os.path.join(out_jpg_dir,folder),form_jpg_path)
    shutil.move(os.path.join(out_xml_dir,folder),form_xml_path)


# In[78]:



def identify_form_type(xml_tree,page_height,out_jpg_dir,out_xml_dir,form_type_path,folder):
    text_list=[]
    for elem in xml_tree.findall('.//page/flow/block/line/word'):
        text_list.append(elem.text)
#         if(float(elem.attrib['yMax'])<(float(page_height)/6)):
#             text_list.append(elem.text)
#     print(text_list)
#     print(xml_tree)
    if ('MV-104A' in text_list or '-104A' in text_list or '104A' in text_list or 'MV04A' in text_list or 'MO4A' in text_list or 'MV-104AN' in text_list or 'MV-104AN(7/11)' in text_list):
        try:
            move_folder(out_jpg_dir,out_xml_dir,form_type_path[0],folder)
            print('Form Type 1')
        except:
            print("Warning file already exist in destination check identify_form_type")
            pass

    else:
        try:
            move_folder(out_jpg_dir,out_xml_dir,form_type_path[1],folder)
            print('Non Crash Form')
        except:
            print("Warning file already exist in destination check identify_form_type")
            pass


# In[79]:
def form_type_detection(out_jpg_dir,out_xml_dir,form_type_path):
#     print('form type detections starts')
    xml_folder_list=os.listdir(out_xml_dir)
#     print('xml folder list is',xml_folder_list)
    for folder in tqdm(xml_folder_list):
        xml_path = os.path.join(out_xml_dir,folder,'*.xml')
        
        xml_list = glob.glob(xml_path)
        xml_list=sorted(xml_list)
        jpg_path = os.path.join(out_jpg_dir,folder,'*.jpg')
        jpg_list = glob.glob(jpg_path)
#         print(xml_list)
        if(jpg_list==None):
            print("JPG corresponding to XML does not exist")
        else:
            try:
                #print(xml_list[0])
                xml_str = open(xml_list[0],'r',encoding="utf-8").read()
                upd_xml_str=str_replace_xml(xml_str)
                xml_tree=ET.XML(upd_xml_str)
                page_list=xml_tree.findall('.//page')
                page_height = page_list[0].attrib['height']
                identify_form_type(xml_tree,page_height,out_jpg_dir,out_xml_dir,form_type_path,folder)
            except Exception as e:
                print(e)
#     shutil.rmtree(out_jpg_dir)
#     shutil.rmtree(out_xml_dir)


# In[80]:
def rec_plot(coord, img, lab):
    xmin,ymin,xmax,ymax = coord
    ann_img=cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax),int(ymax)),(0,255,0),2)
    cv2.putText(ann_img, lab, (int(xmax-10), int(ymax+10)), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    return ann_img


def draw_ann(df):
    img_list = set(df.image_path)
    n = len(img_list)
    for j in tqdm(range(n)):
        gt_df = df.sort_values(by='image_path')
        gt_df = gt_df[~gt_df.classes.str.contains("classes")]
        uniq_name = gt_df.image_path.unique()
        image_path =str(uniq_name[j])
        img = cv2.imread(image_path)
        df_grp = gt_df.groupby("image_path").get_group(str(uniq_name[j]))
        for i, row in df_grp.iterrows():
            xmin = row.x1
            ymin = row.y1
            xmax = row.x2
            ymax = row.y2
            lab = row.classes
            gt_ann_img = rec_plot((xmin,ymin,xmax,ymax), img, lab)
        ann_image= os.path.dirname(image_path)+"\\"+(os.path.basename(image_path).split(".")[0])+"_annot.jpg"
        if not os.path.exists(ann_image):
            cv2.imwrite(ann_image, gt_ann_img)
        else:
            os.remove(ann_image)
            cv2.imwrite(ann_image, gt_ann_img)
# In[81]:
    
def pred_detect(form_type_path,frozen_model_path,label_map):
    form_type_list = os.listdir(form_type_path)
    comp_img_list = []
    for j in form_type_list:
        path = os.path.join(form_type_path,j,'jpg','*.jpg')
        comp_img_list.extend(glob.glob(path))
        
    df=create_detection_df(comp_img_list,frozen_model_path,label_map,score_thresh=0.9)
    
    df.to_csv(os.path.join(form_type_path,'predictions.csv'))
    return(df)


# In[82]:


def update_coord(coord,factor_height):
    coord = [float(i)*factor_height for i in coord]
    return(coord)

def intersection(ocr_coord,bb_coord):
    [y_min1, x_min1, y_max1, x_max1] = np.split(ocr_coord, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(bb_coord, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
      np.zeros(all_pairs_max_ymin.shape),
      all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
      np.zeros(all_pairs_max_xmin.shape),
      all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


# In[83]:



def area(ocr_coord):

    return (ocr_coord[:, 2] - ocr_coord[:, 0]) * (ocr_coord[:, 3] - ocr_coord[:, 1])


def check_text_in_BB(ocr_coord,bb_coord,ocr_text):
    ocr_coord = np.expand_dims(np.array(ocr_coord),axis=0)
    bb_coord = np.expand_dims(np.array(bb_coord),axis=0)
    intersect = intersection(ocr_coord,bb_coord)
    text_area = area(ocr_coord)
    intersect_over_text_area = intersect/text_area
    if (intersect_over_text_area>0.5):
        result_text = ocr_text
    else:
        result_text = ''
    return(result_text)


# In[84]:


#from keras.models import load_model
#from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
#from keras.preprocessing.image import img_to_array
    
#model_path = r"C:\Users\a\Downloads\Extraction\new_model_.h5"
#model = load_model(model_path,compile=False)


# In[85]:


def checking(img,xmin,ymin,xmax,ymax):
    # print('yessssss')
    xmin=int(xmin+4)
    ymin=int(ymin+4)
    xmax=int(xmax-4)
    ymax=int(ymax-4)
    im1=img[ymin:ymax,xmin:xmax] 
    ret,threshold_image = cv2.threshold(im1,127,255,cv2.THRESH_TOZERO)
    counts=cv2.countNonZero(threshold_image)
    width,height=im1.shape
    total_pixels=width*height
    #print(total_pixels)
    if((total_pixels-counts)/total_pixels)>0.1:
        return 'checked'
    else:
        return 'unchecked'
    
# from keras.models import load_model
# from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
# from keras.preprocessing.image import img_to_array
# import datetime    
# model_path = "/home/developer/Extraction/new_model_.h5"
# model = load_model(model_path, compile=False)

# def checking(image,xmin,ymin,xmax,ymax):
#     print("yesssssssssssss")
    
#     label_map=dict({0:"Checked",
#                     1:"Unchecked"})
    
#     image = image[ymin:ymax,xmin:xmax]
#     image = img_to_array(image)
#     image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
#     image = preprocess_input_vgg16(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     y_pred = model.predict(image)
#     y_pred = np.argmax(y_pred, axis=1)
#     y_pred = label_map[y_pred[0]]
#     print('y_pred is',y_pred)
#     return y_pred    


# In[86]:


def find_address_text_y(xml_tree,bb_coord,c):
    add_list = []
    text_list = []
    for elem in xml_tree.findall('.//page/flow/block/line/word'):
        coord=(elem.attrib['xMin'],elem.attrib['yMin'],elem.attrib['xMax'],elem.attrib['yMax'])
        ocr_coord=update_coord(coord,c)
        result_text = check_text_in_BB(ocr_coord, bb_coord, elem.text)
        if(result_text is not None and result_text!=''):
            add_list.append([float(elem.attrib['yMin']),result_text]) 
    return(add_list)


# In[87]:


def find_address_text(xml_tree,bb_coord,c):
    add_list = []
    text_list = []
    for elem in xml_tree.findall('.//page/flow/block/line/word'):
        coord=(elem.attrib['xMin'],elem.attrib['yMin'],elem.attrib['xMax'],elem.attrib['yMax'])
        ocr_coord=update_coord(coord,c)
        result_text = check_text_in_BB(ocr_coord, bb_coord, elem.text)
        if(result_text is not None and result_text!=''):
            add_list.append([float(elem.attrib['xMin']),result_text]) 
    return(add_list)
#
# =============================================================================
# #This is the accident description check works based on the pixel strength for only one filed
# =============================================================================
def accident_description_check(image,temptext_df):
    #This image should be a grayscale image.
    temp=temptext_df.copy()
    ignore_labels=['rear_end','right_turn_1','head_on','left_turn_1','right_angle','right_turn_2','left_turn_2','side_swipe_2','side_swipe_1']
    check_data=temp[temp['label'].isin(ignore_labels)]
    for i,row in check_data.iterrows():
        im=image[int(row.ymin)+5:int(row.ymax)-5,int(row.xmin)+5:int(row.xmax)-5]
        value = (im.shape[0]*im.shape[1])-cv2.countNonZero(im)
        #Threshold set manually based on requirement.
        if(value>1450):
            check_data.loc[i,'text']='checked'
        else:
            check_data.loc[i,'text']='unchecked'
    temp.loc[check_data.index] = np.nan
    temp = temp.combine_first(check_data)
    return temp
# =============================================================================
# #This is the helper fuction fo the third_party_details function
# =============================================================================
def function_of_third_party_details(ymin,temp_df,index):
    #generally index willl be 0,1,2,3,4,5
    diction={}
    count=0
    flag=0
    lis1=['ymin1','ymin2','ymin3','ymin4','ymin5','ymin6','ymin7','ymin8','ymin9','ymin10','ymin11']
    lis2=['field_10','field_11','field_12','field_18','field_15','field_16','field_17','field_9','field_8','accident_involved_person','date_of_death']
    for i,j in zip(lis1,lis2):
        try:
            diction[i]=int(temp_df[temp_df['label']==j].sort_values('ymin').iloc[index,2])
        except:
            diction[i]=0
    for i in diction.values():
        if i-8<int(ymin)<i+8:
            count=count+1
    if(count>4):
        flag=1
    return flag
# =============================================================================
# #This is the funnction for the third party details, to assingning the number based on the row it has a helper function above
# =============================================================================
def third_party_details(dataframe,count_accident_page):
    temp_df=dataframe.copy()
    temp_df1=dataframe.copy()
    #hardcoded based on requirement.
    append_number=count_accident_page
    coordinate_labels=['field_8','field_9','field_10','field_11','field_12','accident_involved_person_gender','field_14','field_15','field_16','field_17','field_18','accident_involved_person','date_of_death']
    for i in coordinate_labels:
        temp_append_number=append_number
        sub_df=temp_df[temp_df['label']==i].sort_values('ymin')
        ymin=sub_df.iloc[0,2]
        for k in range(5):
            value=function_of_third_party_details(ymin,temp_df,k)
            if(value==0):
                temp_append_number=temp_append_number+1
            elif(value==1):
                break
        previous_ymin=0
        for j, row in sub_df.iterrows():
            ymin=temp_df.loc[j,'ymin'] 
            if(previous_ymin<ymin):
                temp_df1.loc[j,'label']=temp_df1.loc[j,'label']+'_in_row_'+str(temp_append_number)
            elif(previous_ymin>ymin):
                temp_df1.loc[j,'label']=temp_df1.loc[j,'label']+'_in_row_'+str(temp_append_number+1)
            previous_ymin=ymin
            temp_append_number=temp_append_number+1
        temp_append_number=0
    count_accident_page=count_accident_page+6 #since there are 6 rows in each page.
    return temp_df1,count_accident_page
# In[88]:
# =============================================================================
# This is the function for marking all as unchecked if more than one related field is marked as checked.
# =============================================================================
def remove_more_than_one_checked_fields(das):
    diction={}
    diction['l1']=['police_photos_yes','police_photos_no']
    diction['l2']=['county_of_crash_village', 'county_of_crash_town','county_of_crash_city']
    #diction['l1']=['vehicle_width_>34','vehicle_width_>95','vehicle_overdimension_permit','vehicle_overweight_permit']
    diction['l3']=['other_pedestrian','pedestrian','bicyclist','vehicle']
    diction['l4']=['direction_east', 'direction_north', 'direction_south', 'direction_west']
    diction['l5']=['crash_bronx','crash_kings','crash_newyork','crash_queens','crash_richmond']
    diction['l6']=['cost_of_repair_<1000', 'cost_of_repair_>1000', 'cost_of_repair_unable_to_determine']
    diction['l7']=['highway_dist_no', 'highway_dist_yes']
    diction['l8']=['accident_equipment_4way_flasher', 'accident_equipment_headlights', 'accident_equipment_horn','accident_equipment_siren',
     'accident_equipment_traffic_cones', 'accident_equipment_turret_light', 'accident_equipment_warning_lights']
    diction['l9']=['police_action_code_signal','police_action_comply_station_house','police_action_other','police_action_pursuing_violator'
     ,'police_action_routine_patrol']
    for i in diction.keys():
        das1=das[das['label'].isin(diction[i])]
        try:
            if(das1['text'].value_counts().checked)>1:
                das1.loc[:,'text']='unchecked'
                das.loc[das1.index] = np.nan
                das = das.combine_first(das1)
        except:
            continue
    return das
    #Including the try catch because if the all boxesa are un checked
# =============================================================================
# #This function is for splitting the parties based on the party id. like driver1 driver 2 if it is from different page driver 3 driver 4 so on.
# =============================================================================
def split_fields_based_on_parties_accident_page(temptext_df,coordinate_labels,accident_page_count):
    temptext_df_copy=temptext_df.copy()
    for i in coordinate_labels:
        append_number=2*accident_page_count-1
        sub_df=temptext_df_copy[temptext_df_copy['label']==i].sort_values('xmax')
        if(sub_df.shape[0]==2):
            for j, row in sub_df.iterrows():
                temptext_df_copy.loc[j,'label']=temptext_df_copy.loc[j,'label']+'_belongs_to_'+str(append_number)
                append_number=append_number+1 
        if(sub_df.shape[0]==1):
            max_xmax=max(temptext_df_copy['xmax'])
            min_xmin=min(temptext_df_copy['xmin'])
            if int(sub_df['xmin'])<((min_xmin+max_xmax)/2):
                for j, row in sub_df.iterrows():
                    temptext_df_copy.loc[j,'label']=temptext_df_copy.loc[j,'label']+'_belongs_to_'+str(1)
            elif int(sub_df['xmin']+20)>((min_xmin+max_xmax)/2):
                for j, row in sub_df.iterrows():
                    temptext_df_copy.loc[j,'label']=temptext_df_copy.loc[j,'label']+'_belongs_to_'+str(2)
    accident_page_count=accident_page_count+1
    return temptext_df_copy,accident_page_count
    
# =============================================================================
# #This is the logic for the splitting the fields based on the page2 for the  based on the coordinates
# ["accident_person_name","accident_person_address","date_of_birth","accident_person_telephone"]
# =============================================================================
def persons_invloved_in_accident_second_page(data):
    l1=["accident_person_name","accident_person_address","date_of_birth","accident_person_telephone"]
    data1=data[data.label.isin(l1)]
    mid=(min(data['xmin'])+max(data['xmax']))/2
    data1_left=data1.loc[(data1['xmin'] < mid) & (data1['xmax'] < mid+300)].sort_values('label')
    #data1_left[data1_left['label']=="date_of_birth"]['ymin'].values.tolist()
    for i in l1:
        temp_df=data1_left[data1_left['label']==i].sort_values('ymin')
        if(temp_df.shape[0]==3):
            count=1
            for j, row in temp_df.iterrows():
                data1_left.loc[j,'label']=data1_left.loc[j,'label']+'_'+str(count)
                count=count+1
    data1_right=data1.loc[(data1['xmin'] > mid-200) & (data1['xmax'] > mid)].sort_values('label')
    for i in l1:
        temp_df=data1_right[data1_right['label']==i].sort_values('ymin')
        if(temp_df.shape[0]==2):
            count=4
            for j, row in temp_df.iterrows():
                data1_right.loc[j,'label']=data1_right.loc[j,'label']+'_'+str(count)
                count=count+1
    data.loc[data1_left.index] = np.nan
    data = data.combine_first(data1_left)
    data.loc[data1_right.index] = np.nan
    data = data.combine_first(data1_right)
    return data
# =============================================================================
# """This is the logic for the splitting the fields in the second page based on the coordinates
#for the fields 'property_damaged','property_owner','notification_details','vehicle_number','registration_expires','vin' """
# =============================================================================
def split_fields_page_2(df,page_2_count):
    data=df.copy()
    lis1=['property_damaged','property_owner','notification_details']
    lis2=['vehicle_number','registration_expires','vin']
    data1=data[data['label'].isin(lis1)]
    for i in lis1:
        append_number1=2*page_2_count-1
        if(len(data1[data1['label']==i])>0):
            temp_df=data1[data1['label']==i].sort_values('ymin')
            for j, row in temp_df.iterrows():
                data.loc[j,'label']=data.loc[j,'label']+'_belongs_to_'+str(append_number1)
                append_number1=append_number1+1
        else:
            continue
    data2=data[data['label'].isin(lis2)]
    for i in lis2:
        append_number2=2*page_2_count-1
        if(len(data2[data2['label']==i])>0):
            temp_df=data2[data2['label']==i].sort_values('xmin')
            for j, row in temp_df.iterrows():
                data.loc[j,'label']=data.loc[j,'label']+'_belongs_to_'+str(append_number2)
                append_number2=append_number2+1
        else:
            continue
    return data,page_2_count+1
# =============================================================================
# """this is the logic for the vehicle_by and towed_to fields in the image"""           
# =============================================================================
def split_vehicle_by_towed_to(dataframe):
    data=dataframe.copy()
    lis=["vehicle_by","towed_to"]
    for i in lis:
        data1=data[data['label']==i]
        if(data1.shape[0]==2):
            count=1
            data1=data1.sort_values("xmin")
            for j,row in data1.iterrows():
                data.loc[j,'label']=data.loc[j,'label']+"_"+str(count)
                count=count+1
        else:
            continue
    return data
# =============================================================================
# Not creating the dataframe for the pages with no information.
# =============================================================================
def remove_the_extra_pages_that_are_not_required(dataframe):
    data=dataframe.copy()
    labels=list(data.label)
    labels_to_be_subset=['driver_address',"driver_apt_no",'driver_date_of_birth','driver_details','driver_license_no','driver_name',
'driver_sex','driver_state','driver_unlicensed','owner_address','owner_address_details','owner_apt_no','owner_date_of_birth','owner_name']
    c = sum(el in labels for el in labels_to_be_subset)
    if c>5:
        return True
    else:
        return False
# =============================================================================
# #Function for the party_id_unit_number appending the party id or the unit number
# =============================================================================
def function_for_party_id_unit_number(dataframe,party_count):
    data=dataframe.copy()
    data1=data[data['label']=="vehicle_id"]
    data1 = pd.concat([data1]*2, axis=0)
    data1=data1.reset_index(drop=True)
    append_number=2*party_count-1
    for i,row in data1.iterrows():
        data1.loc[i,'label']=data1.loc[i,'label']+'_belongs_to_'+str(append_number)
        append_number=append_number+1
    party_count=party_count+1
    data=pd.concat([data, data1],axis=0)
    return data,party_count
# =============================================================================
# #Logic for the incident extraction appending the only for the page1
# =============================================================================
def function_for_incident_extraction(datax):
    data=datax.copy()
    incident_page_list=['crash_date','crash_no','crash_id','place/city_of_crash','crash_occured','intersection_with_street_road_highway','at_latitude',
    'at_longitude','left_scene','police_photos','area_7','area_6','beat','zone_number','area_28','area_4','area_3',
    'left_turn_1','left_turn_2','right_angle','right_turn_1','right_turn_2','side_swipe_1','side_swipe_2']
    data1=data[data['label'].isin(incident_page_list)]
    data1.loc[:,'label']=data1.loc[:,'label']+'_present_in _page_1'
    data.loc[data1.index] = np.nan
    data = data.combine_first(data1)
    return data
# =============================================================================
# #This is the final text extraction function which finally generates the csv
# =============================================================================
def final_text_extract(df,check_box_labels,ignore_check,numerical_labels,coordinate_labels):
    df = df.sort_values(["y1", "x1"], ascending = (True, True))
    group_df = df.groupby('image_path')
    img_list = sorted(df['image_path'].unique())
    accident_page_count=1
    party_count=1
    count_accident_page=1
    page_2_count=1
    temptext_df_final=pd.DataFrame()
    for img in tqdm(img_list):
        image=cv2.imread(img)
        image1=cv2.imread(img,0)
        height,width,channels=image.shape
        xml_name=img.replace('jpg','xml')
        xml_str = open(xml_name,'r',encoding='utf-8').read()
        upd_xml_str=str_replace_xml(xml_str)
        xml_tree=ET.XML(upd_xml_str)
        page = open(xml_name,encoding="utf-8")
        soup = BeautifulSoup(page.read(),'html.parser')
        page=soup.findAll('page')
        for tags in page:
            width_xml=float(tags.get('width'))
            height_xml=float(tags.get('height'))
        factor_height=height/height_xml
        factor_width=width/width_xml
        if True:
            temp_df=group_df.get_group(img)
            csv_name = img.split('.')[0]+'.csv'
            data_list=[]
            for i,row1 in temp_df.iterrows():
                bb_xmin = int(row1.x1)
                bb_ymin = int(row1.y1)
                bb_xmax = int(row1.x2)
                bb_ymax = int(row1.y2)
                text_list = []
                text=''
                #print(row1.classes)
                if(row1.classes in check_box_labels):
                    text=checking(image1,row1.x1,row1.y1,row1.x2,row1.y2)
                    data_list.append((row1.image_path,row1.x1,row1.y1,row1.x2,row1.y2,row1.classes,str(text)))
                    continue
                if(row1.classes in ignore_check):
                    text='ignore'
                    data_list.append((row1.image_path,row1.x1,row1.y1,row1.x2,row1.y2,row1.classes,str(text)))
                    continue
                for elem in xml_tree.findall('.//page/flow/block/line/word'):
                    coord=(elem.attrib['xMin'],elem.attrib['yMin'],elem.attrib['xMax'],elem.attrib['yMax'])
                    ocr_coord=update_coord(coord,factor_height)
                    bb_coord=(bb_xmin,bb_ymin,bb_xmax,bb_ymax)
                    result_text = check_text_in_BB(ocr_coord, bb_coord, elem.text)
                    text_list.append(result_text)
                if(len(text_list)==0):
                    text=''
                else:
                    try:
                        text=' '.join(text_list)
                        if(row1.classes!='narration'):
                           text=re.sub(r'^[\W\.\_]*','',text)
                           text=re.sub(r'\W*$','',text)
                           if((re.match("[a-zA-Z0-9]*", text))==None):
                              text='' 
                           text=re.sub(r"(\d+)\s(-)\s(\d+)",r"\1\2\3",text.rstrip())
                           text=re.sub(r"(\d+)(-)\s(\d+)",r"\1\2\3",text.rstrip())
                           text=re.sub(r"(\d+)\s(-)(\d+)",r"\1\2\3",text.rstrip())
                           text=re.sub(r"(\d+)\s(\d+)",r"\1\2",text.rstrip())
                           text=re.sub(' +', ' ', text)
                           char_check_end = re.compile('[(]')
                           if(char_check_end.search(text)!=None):
                               text=text+')'
                           char_check_beg = re.compile('[)]')
                           if(char_check_beg.search(text)!=None):
                               if(char_check_end.search(text)==None):
                                   text='('+text
                           if(text=='I'):
                               text = '1'
                        if(row1.classes.startswith('vehicle_damaged_codes')):
                            add_list = find_address_text_y(xml_tree,bb_coord,factor_height)
                            add_list = sorted(add_list,key= lambda x:x[0])
                            text_list = [x[1] for x in add_list] 
                            text=' '.join(text_list)
                            fine=re.findall(r'[0-9]+',text)
                            x=str(row1.classes)[-1]
                            if(x==fine[0]and len(fine)==2):
                                text=fine[1]
                            if(x==fine[0] and len(fine)==1):
                                text=''
                            if(x!=fine[0] and len(fine)==1):
                                text=fine[0]
                            if(x==fine[0]and len(fine)==3):
                                text=fine[1:2]
                        if(row1.classes.startswith('driver_details')or row1.classes.startswith('owner_address')or row1.classes.startswith('reporting_officer_rank')or row1.classes.startswith('reporting_officer_name')or row1.classes.startswith('driver_name') or row1.classes.startswith('owner_name') or row1.classes.startswith('owner_address_details') or row1.classes.startswith('driver_address')):
                            
                            add_list = find_address_text(xml_tree,bb_coord,factor_height)
                            add_list = sorted(add_list,key= lambda x:x[0])
                            text_list = [x[1] for x in add_list]   
                            text=''
                            for j in range(len(text_list)):
                                 text =  text + text_list[j] + ' '
                            text=re.sub(' +', ' ', text)
                            

                        #if(row1.classes.startswith('crash_date') or row1.classes.startswith('date_of_birth') or row1.classes.startswith('driver_date_of_birth') or row1.classes.startswith('owner_date_of_birth')):
                        if(row1.classes.startswith(('crash_date','date_of_birth','driver_date_of_birth','owner_date_of_birth'))):
                           add_list = find_address_text(xml_tree,bb_coord,factor_height)
                           add_list1 = sorted(add_list)
                           text=''
                           for j in range(len(add_list1)):
                                text =  text + add_list1[j][1] + '/'
                           if len(text)>0 and text[-1]=='/':
                               text=text[:-1]


                        if(row1.classes.startswith('reporting_officer_name')):
                            try:
                                if text.startswith('in '):
                                    text=text[2:]
                            except:
                                pass
                        try:
                            if(row1.classes.startswith('at_longitude') or row1.classes.startswith('at_latitude')):
                                text=' '.join(text_list)
                                text=re.sub(' +', '', text)
                                if(float(text)>180):
                                    text=text[:2]+'.'+text[2:]
                        except:
                            pass
                        if(row1.classes in numerical_labels):
                            text = ''.join(re.findall(r'\d+', text))
                        if(row1.classes =='intersection_with_street_road_highway' or row1.classes =='crash_occured'):
                            words = ['Route', 'Number','or','Name']
                            for word in words:
                                text = text.replace(word, '')

                        if(row1.classes.startswith('field_')):
                            text=re.sub(r'II*', "", text)
                            text=re.sub(r'ill*', "", text)
                            text=re.sub(r'll*', "", text)
                        if(row1.classes.startswith('reporting_officer_rank')):
                            pattern=r"[0-9\.\,\-\|\<\*\'\"\"'\?']"
                            pattern1=re.compile(pattern)
                            text=re.sub(pattern1,'',text)
                            if(text.startswith('PATROLMAN')):
                                text='PATROLMAN'
                            if(text.startswith('POLICE OF')):
                                text='POLICE OF'
                            if(text.startswith('POM')):
                                text='POM'
                            if(text.startswith('TPR')):
                                text='TPR'
                        if(row1.classes.startswith('intersection_with_street_road_highway_another')):
                           if(text.startswith('of ')):
                               text=text[3:]
                        if(row1.classes.startswith('accident_person_name')):
                            words = ['M.I', 'First','Last Name']
                            for word in words:
                                text = text.replace(word, '')
                        if text.startswith(' '):
                            text=text[1:]
                        text=re.sub(r'\. \.', " ", text)
                        text=re.sub(r'\.\.', " ", text)
                        text=re.sub(r'\._', "", text)
                        text=re.sub(r'\s+I', "", text)
                        text=re.sub(r'/-/', "/", text)
                        text=re.sub(r"-'", "", text)
                        text=re.sub(r'By: ','',text)
                        text=re.sub(r'To: ','',text)
                        if(row1.classes=='narration'):
                            words = ['Accident', "Description/Officer's",'Notes','notes']
                            for word in words:
                                text = text.replace(word, '')
                            text=re.sub(' +', ' ', text)
                            text=text.strip()
                    except :
                        text=''
                data_list.append((row1.image_path,row1.x1,row1.y1,row1.x2,row1.y2,row1.classes,str(text)))
            temptext_df = pd.DataFrame(data_list,columns=['path','xmin','ymin','xmax','ymax','label','text'])
            labels_in_accident_page=list(temptext_df['label'])
            try:
                sub_list=['day_of_week','crash_no','driver_unlicensed','crash_id','scene_investigation']
                if(set(labels_in_accident_page)&(set(sub_list))):
                    print("final_text_extract -> remove_the_extra_pages_that_are_not_required started")
                    flag=remove_the_extra_pages_that_are_not_required(temptext_df)
                    if flag==True:
                        pass
                    elif flag==False:
                        continue
            except:
                print("Error in final_text_extract -> remove_the_extra_pages_that_are_not_required")
                
            try:
                print("accident_description_check has started")
                temptext_df=accident_description_check(image1,temptext_df)
            except:
                print("Error in final_text_extract -> accident_description_check")
            try:
                print('remove_more_than_one_checked_fields started')
                temptext_df=remove_more_than_one_checked_fields(temptext_df)
            except:
                print('Error in final_text_extract -> remove_more_than_one_checked_fields ')
            try:
                sub_list=['area_1','area_2','area_3']
                if((set(sub_list).issubset(set(labels_in_accident_page)))and accident_page_count==1):
                    print('function_for_incident_extraction started')
                    temptext_df=function_for_incident_extraction(temptext_df)
            except:
                print("Error in final_text_extract -> function_for_incident_extraction")
            try:
                sub_list=['area_1','area_2','area_3']
                if(set(sub_list).issubset(set(labels_in_accident_page))):
                    print('split_fields_based_on_parties_accident_page started')
                    temptext_df,accident_page_count=split_fields_based_on_parties_accident_page(temptext_df,coordinate_labels,accident_page_count)
                    #csv_name1 = img.split('.')[0]+'_names_splitted.csv'
                    #temptext_df1.to_csv(csv_name1,index=False)
            except:
                print("Error in final_text_extract -> split_fields_based_on_parties_accident_page")
            try:
                sub_list=['area_1','area_2','area_3']
                if(set(sub_list).issubset(set(labels_in_accident_page))):
                    print("final_text_extract -> third_party_details started")
                    temptext_df,count_accident_page=third_party_details(temptext_df,count_accident_page)
            except:
                print("Error in final_text_extract -> third_party_details")
            try:
                sub_list=['area_1','area_2','area_3']
                if(set(sub_list).issubset(set(labels_in_accident_page))):
                    print("final_text_extract -> split_vehicle_by_towed_to started")
                    temptext_df=split_vehicle_by_towed_to(temptext_df)
            except:
                print("Error in final_text_extract -> split_vehicle_by_towed_to")
            try:
                sub_list=["accident_person_name","accident_person_address","date_of_birth","accident_person_telephone"]
                if(set(sub_list).issubset(set(labels_in_accident_page))):
                    print("final_text_extract -> persons_invloved_in_accident_second_page started")
                    temptext_df=persons_invloved_in_accident_second_page(temptext_df)
            except:
                print("Error in final_text_extract -> persons_invloved_in_accident_second_page")                
            try:
                sub_list=["accident_person_name","accident_person_address","date_of_birth","accident_person_telephone"]
                if(set(sub_list).issubset(set(labels_in_accident_page))):
                    print("final_text_extract -> split_fields_page_2 started")
                    temptext_df,page_2_count=split_fields_page_2(temptext_df,page_2_count)
            except:
                print("Error in final_text_extract -> split_fields_page_2")
            
            try:
                sub_list=['area_1','area_2','area_3']
                if(set(sub_list).issubset(set(labels_in_accident_page))):
                    print('function_for_party_id_unit_number started')
                    temptext_df,party_count=function_for_party_id_unit_number(temptext_df,party_count)
            except:
                print("Error in final_text_extract -> function_for_party_id_unit_number")
            temptext_df.to_csv(csv_name,index=False)
            print('csv is created')
            temptext_df_final=temptext_df_final.append(temptext_df, ignore_index = True)
    temptext_df_final.to_csv(img.split('.')[0]+'_final.csv' )

# In[90]:


def ecrash_form_split(img_path,pdf_path,out_jpg_dir,out_xml_dir,form_type_path,frozen_model_path,label_map):
    try:
        print("creation of directories")
        create_dirs(out_jpg_dir,out_xml_dir,form_type_path)
    except:
        print("exception occured in create_dirs" )
    img_list = glob.glob(img_path+'*.tif')
    for i in img_list:
        try:
            print("tiff_to_jpg conversion started")
            tiff_to_jpg(i,out_jpg_dir)
        except:
            print("Exception occured in tiff_to_jpg" )
    try:
        print("xml_convertion has started")
        xml_convertion(pdf_path,out_xml_dir)
    except:
        print("There is exception in xml_convertion function")
    try:
        print("form_type_detection Starts")
        form_type_detection(out_jpg_dir,out_xml_dir,form_type_path)
    except:
        print("Error in form_type_detection function")
####################################################################################################################
#hardcoded    
#These are check box labels
    check=pd.read_csv(r"C:\Users\a\Downloads\Extraction\check_boxes_labels.csv")
    check_box_labels=list(check['uique_labels'])
    #This ignore labels are ignored conists of the left swipe , right swipe etc...
    ignore=pd.read_csv(r"C:\Users\a\Downloads\Extraction\ignorelabels.csv")
    ignore_check=list(ignore['ignore_labels'])
    #This numericcal label are labels which contain only the numericals in case of the alphbetical extraction we can remove
    num=pd.read_csv(r"C:\Users\a\Downloads\Extraction\numerical_labels.csv")
    numerical_labels=list(num['numerical_labels'])
    #This coordinate labels are based on the more than one party.
    coordinate_data=pd.read_csv(r"C:\Users\a\Downloads\Extraction\coordinate_labels.csv")
    coordinate_labels=list(coordinate_data['coordinate_labels'])
    
######################################################################################################################
    for j in range(1):
        try:
            print("Done")
            df=pred_detect(form_type_path[j],frozen_model_path,label_map)
        except:
            print("Exception occured in pred_detect function")
        try:
            df=pd.read_csv(os.path.join(form_type_path[j],'predictions.csv'))
            draw_ann(df)
        except:
            print("Exception occured in the draw_ann")
        try:
            final_text_extract(df,check_box_labels,ignore_check,numerical_labels,coordinate_labels)
        except:
            print("Exception occured in the final_text_extract")
       
    

pdf_path = r"C:\Users\a\Downloads\Extraction\DATA1\PDF\\"
img_path =  r"C:\Users\a\Downloads\Extraction\DATA1\TIF\\"

out_jpg_dir = r"C:\Users\a\Downloads\Extraction\DATA1\OUTPUT\JPG\\"
out_xml_dir = r"C:\Users\a\Downloads\Extraction\DATA1\OUTPUT\XML\\"

form_type_path = [r"C:\Users\a\Downloads\Extraction\DATA1\OUTPUT\Form1\\",r"C:\Users\a\Downloads\Extraction\DATA1\OUTPUT\Form_not_valid\\"]
frozen_model_path = r"C:\Users\a\Downloads\Extraction\frozen_inference_graph.pb"
label_map = r"C:\Users\a\Downloads\Extraction\label_map.pbtxt"

ecrash_form_split(img_path,pdf_path,out_jpg_dir,out_xml_dir,form_type_path,frozen_model_path,label_map)