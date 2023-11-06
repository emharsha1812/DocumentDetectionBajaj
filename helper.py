from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
from io import BytesIO
from pdf2image import convert_from_path
import settings


def converttoimage(pdf_path):
    pdf_images = convert_from_path(pdf_path)
    for idx in range(len(pdf_images)):
         pdf_images[idx].save('pdf_page_'+ str(idx+1) +'.png', 'PNG')
    return pdf_images[0]


        
   
    
    
    





#Downloadable format helper function
def downloadIt(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return byte_im


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


# def play_rtsp_stream(conf, model):
#     """
#     Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

#     Parameters:
#         conf: Confidence of YOLOv8 model.
#         model: An instance of the `YOLOv8` class containing the YOLOv8 model.

#     Returns:
#         None

#     Raises:
#         None
#     """
#     source_rtsp = st.sidebar.text_input("rtsp stream url:")
#     st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
#     is_display_tracker, tracker = display_tracker_options()
#     if st.sidebar.button('Detect Objects'):
#         try:
#             vid_cap = cv2.VideoCapture(source_rtsp)
#             st_frame = st.empty()
#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()
#                 if success:
#                     _display_detected_frames(conf,
#                                              model,
#                                              st_frame,
#                                              image,
#                                              is_display_tracker,
#                                              tracker
#                                              )
#                 else:
#                     vid_cap.release()
#                     # vid_cap = cv2.VideoCapture(source_rtsp)
#                     # time.sleep(0.1)
#                     # continue
#                     break
#         except Exception as e:
#             vid_cap.release()
#             st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


# def play_uploaded_video(uploaded_file,conf,model):
    
#     """
#     Plays the uploaded video. Tracks and detects objects using YOLOv8 
#     """
#     input_path = uploaded_file.name
#     file_binary = uploaded_file.read()
    
#     if st.sidebar.button('Detect Video Objects'):
#         try:
#             vid_cap = cv2.VideoCapture(
#                 str(settings.VIDEOS_DICT.get(source_vid)))
#             st_frame = st.empty()
#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()
#                 if success:
#                     _display_detected_frames(conf,
#                                              model,
#                                              st_frame,
#                                              image,
#                                              is_display_tracker,
#                                              tracker
#                                              )
#                 else:
#                     vid_cap.release()
#                     break
#     except Exception as e:
#             st.sidebar.error("Error loading video: " + str(e))
    
#     # with open(input_path, "wb") as temp_file:
#     #     temp_file.write(file_binary)
#     #     video_stream = cv2.VideoCapture('video.mp4')
        
#     # with st.spinner('Processing video...'): 
#     #     while True:
#     #         ret, frame = video_stream.read()
#     #         if not ret:
#     #             break
#     #         result = model(frame)
#     #         for detection in result[0].boxes.data:
#     #             x0, y0 = (int(detection[0]), int(detection[1]))
#     #             x1, y1 = (int(detection[2]), int(detection[3]))
#     #             score = round(float(detection[4]), 2)
#     #             cls = int(detection[5])
#     #             object_name =  model.names[cls]
#     #             label = f'{object_name} {score}'

#     #             if model.names[cls] in selected_objects and score > min_confidence:
#     #                 cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
#     #                 cv2.putText(frame, label, (x0, y0 - 10),
#     #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
#     #         detections = result[0].verbose()
#     #         cv2.putText(frame, detections, (10, 10),
#     #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     #     video_stream.release()


def count_objects(predictions, target_classes):
    object_counts = {x: 0 for x in target_classes}
    for prediction in predictions:
        for c in prediction.boxes.cls:
            c = int(c)
            if c in target_classes:
                object_counts[c] += 1
            elif c not in target_classes:
                object_counts[c] = 1

    present_objects = object_counts.copy()

    for i in object_counts:
        if object_counts[i] < 1:
            present_objects.pop(i)

    return present_objects


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
