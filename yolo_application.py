import streamlit as st
import os
import cv2
from ultralytics import YOLO
import numpy as np

st.title("YOLO image and vedio Processing.")

# Allow users to upload images or videos
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

# Load the model with giving the weights in which we ran our model in the notebook

model=YOLO('D:\\vscode\\Automatic-Car-Numberplate-Recognition-System\\best.pt')


#Error Handelling
def process_media(input_path,output_path):
    file_extension= os.path.splitext(input_path)[1].lower()
    if file_extension in ['.mp4','.mkv']:
        pass
    elif file_extension in ['.jpeg','.jpg','.png']:
        return predict_and_save_image(input_path, output_path)
    else:
        st.error(f"unsupported file type: {file_extension}")
        return None


# Prediction function for the images will take the image , will make the predictions and save the image and returns the output folder 
def predict_and_save_image(path_test_car, output_image_path):
    results= model.predict(path_test_car,device='cpu')
    image=cv2.imread(path_test_car)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # Conver the image into the RGB format deafault format is BGR
    
    # Now we need to format the predicts stored inside the result
    for result in results:
        for box in result.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0]) #This is the box in XYXY format: [left top coordinates x1,y1, right bottom coordinates x2,y2]. inside the results boxes
            confidence=box.conf[0]
            #now we will pass this coordinates and plot the box 
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            # now we will write confidence score on the image with custom font
            cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert to default format
    cv2.imwrite(output_image_path, image)
    return output_image_path



# here we are saving the uploaded image in the temp directory 
if uploaded_file is not None:
    input_path=f"temp/{uploaded_file.name}"
    output_path=f"output/{uploaded_file.name}"
    with open(input_path,'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.write("Processing Image....")

    result_path = process_media(input_path, output_path)
    if result_path:
        if input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_file = open(result_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
        else:
            st.image(result_path)
 
    

