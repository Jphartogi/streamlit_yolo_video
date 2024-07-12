import cv2
import os
import streamlit as st
from ultralytics import YOLO
import uuid

def app():
    st.header('Object Detection Web App')
    st.subheader('Powered by YOLOv8')
    st.write('Welcome!')
    model = YOLO('yolov8n.pt')
    object_names = list(model.names.values())

    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload video", type=['mp4'])
        selected_objects = st.multiselect('Choose objects to detect', object_names, default=['person']) 
        min_confidence = st.slider('Confidence score', 0.0, 1.0)
        st.form_submit_button(label='Submit')
            
    if uploaded_file is not None: 
        # Generate unique filenames to avoid conflicts
        unique_id = str(uuid.uuid4().hex)[:8]  # Get first 8 characters of UUID for filename
        input_path = os.path.join(os.getcwd(), f"temp_{unique_id}.mp4")
        output_path = os.path.join(os.getcwd(), f"output_{unique_id}.mp4")

        try:
            with open(input_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            video_stream = cv2.VideoCapture(input_path)
            width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)) 
            height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            fourcc = cv2.VideoWriter_fourcc(*'h264') 
            fps = int(video_stream.get(cv2.CAP_PROP_FPS)) 

            out_video = cv2.VideoWriter(output_path, int(fourcc), fps, (width, height)) 

            with st.spinner('Processing video...'): 
                while True:
                    ret, frame = video_stream.read()
                    if not ret:
                        break
                    result = model(frame)
                    for detection in result[0].boxes.data:
                        x0, y0 = (int(detection[0]), int(detection[1]))
                        x1, y1 = (int(detection[2]), int(detection[3]))
                        score = round(float(detection[4]), 2)
                        cls = int(detection[5])
                        object_name =  model.names[cls]
                        label = f'{object_name} {score}'

                        if model.names[cls] in selected_objects and score > min_confidence:
                            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                            cv2.putText(frame, label, (x0, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        else:
                            continue
                    
                    detections = result[0].verbose()
                    cv2.putText(frame, detections, (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                    out_video.write(frame) 
            video_stream.release()
            out_video.release()

            # Delete temporary files after processing
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                st.video(output_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()
