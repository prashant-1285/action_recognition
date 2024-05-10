import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(parentdir)
sys.path.insert(0, parentdir) 

import demo_skeleton_app
# Function to perform inference on the uploaded video
def run_inference(video_file,model_name):
    # Load your deep learning model and perform inference here
    # For demonstration purposes, let's just display a message
  
    st.write("Running inference on the video...will take time")

    # Namespace(checkpoint='https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth',
    #            config='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py', 
    #            det_checkpoint='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth', 
    #            det_config='demo/faster_rcnn_r50_fpn_1x_coco-person.py', 
    #            det_score_thr=0.9,
    #              device='cuda:0', 
    #              label_map='tools/data/label_map/nturgbd_120.txt', 
    #              out_filename='penalty2_out.mp4', 
    #              pose_checkpoint='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth', 
    #              pose_config='demo/hrnet_w32_coco_256x192.py', 
    #              short_side=480, 
    #              video='penalty2.mp4',
    #              out_filename="penalty2_out.mp4")


    video_out = "output.mp4"
    checkpoint_path, label_map_path, config_file_path = select_model(model_name)
    demo_skeleton_app.main(video_file.name, video_out,config_file_path, checkpoint_path, label_map_path )
    st.video(video_out)


def select_model(model_name):
    models = {
        "ucf": {
            "checkpoint_path": "https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ucf101_k400p/s1_joint.pth",
            "label_map_path": "./tools/data/label_map/ucf101.txt",
            "config_file_path": "./configs/posec3d/slowonly_r50_ucf101_k400p/s1_joint.py"
        },
        "k400": {
            "checkpoint_path": "https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_346_k400/joint.pth",
            "label_map_path": "./tools/data/label_map/k400.txt",
            "config_file_path": "./configs/posec3d/slowonly_r50_346_k400/joint.py"
        },
        "gym": {
            "checkpoint_path": "https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_gym/joint.pth",
            "label_map_path": "./tools/data/label_map/gym.txt",
            "config_file_path": "./configs/posec3d/slowonly_r50_gym/joint.py"
        },
        "ntu400": {
            "checkpoint_path": "https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth",
            "label_map_path": "./tools/data/label_map/nturgbd_120.txt",
            "config_file_path": "./configs/posec3d/slowonly_r50_ntu120_xsub/joint.py"
        }
    }
    return models[model_name]["checkpoint_path"], models[model_name]["label_map_path"], models[model_name]["config_file_path"]

# Streamlit App
def main():
    st.title("Deep Learning  Action Recognition Inference")

    # Upload Video Button
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Display the uploaded video on the left side
        
        
        st.video(uploaded_file)

        # Select model
        model_name = st.radio("Select Model", ("ucf", "k400", "gym", "ntu400"))

        # Button to run inference
        if st.button("Run Inference"):
            # Display the inference video on the right side
    
            run_inference(uploaded_file, model_name)

if __name__ == "__main__":
    main()