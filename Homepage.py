# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
import numpy as np
from PIL import Image




# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


counter=0


# Main page heading
st.title(":blue[Document Tampering Detection 🗃️]")

# Sidebar
st.sidebar.header("Configure your Model 🛠️")

# Model Options
# model_type = st.sidebar.radio(
#     "Select Task", ['Detection', 'Segmentation'])

model_type=st.sidebar.radio("Select Task",['Detection 🕵🏻'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence 😎", 0.0, 1.0, 0.40))



# Selecting Detection Or Segmentation
if model_type == 'Detection 🕵🏻':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)




# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)


object_names = list(model.names.values())
# selected_objects = st.sidebar.multiselect('Choose objects to detect', object_names, default=['Whitener'])
container = st.container()
all=st.checkbox('Select all')
select_these=None
if all:
    selected_objects=container.multiselect("Select one or more options:",object_names,object_names)
    select_these=selected_objects
    selected_indices = [object_names.index(option) for option in select_these]
else:
    selected_objects =  container.multiselect("Select one or more options:",
        object_names)
    select_these=selected_objects
    selected_indices = [object_names.index(option) for option in select_these]
    
# st.write(select_these)
source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence,classes=selected_indices
                                    )
                # st.write(class_counts= helper.count_objects(res, model.names))
                # st.write(model.predict(classes=select_these[::]))
                # res=model(uploaded_image,conf=confidence)
                # for detection in res[0].boxes.data:
                #     x0,y0=(int(detection[0]),int(detection[1]))
                #     x1,y1=(int(detection[2]),int(detection[3]))
                #     score=int(detection[5])
                #     cls=int(detection[5])
                #     object_name=model.names[cls]
                #     label=f'{object_name} {score}'
                    
                #     if model.names[cls] in select_these and score>confidence:
                #          boxes = res[0].boxes
                #          res_plotted = res[0].plot()[:, :, ::-1]
                #          st.image(res_plotted, caption='Detected Image',
                #           use_column_width=True)
                    
                
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                # image_array=np.array(res_plotted,'RGB', dtype=np.uint8)
                # final_image=Image.fromarray(image_array)
                # final_image.save("final_image.png")
                # st.write(res_plotted)
                final_image = Image.fromarray(res_plotted)
                filename = f'output{counter}.png'
                final_image.save(filename)
                counter+=1
                with open("output.png", "rb") as file:
                    
                    btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name=filename,
                    mime="image/png"
          )
                # final_image=helper.downloadIt(ff)
                # st.download_button("Download Results",final_image,file_name="detected image")
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")
  

# elif source_radio == settings.VIDEO:
#     uploaded_file=st.file_uploader("Upload Video: ",type=["mp4","mpeg"])
#     if uploaded_file is not None:
#         helper.play_uploaded_video(uploaded_file,confidence,model)
#     else:
#         helper.play_stored_video(confidence, model)

elif source_radio==settings.VIDEO:
    helper.play_stored_video(confidence,model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

# elif source_radio == settings.RTSP:
#     helper.play_rtsp_stream(confidence, model)

# elif source_radio == settings.YOUTUBE:
#     helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
