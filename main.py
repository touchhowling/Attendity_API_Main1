import cv2
import numpy as np
import face_recognition
from fastapi import FastAPI, Response
import uvicorn
import nest_asyncio
import requests
import json

def read_image_from_url(url:str):
    # Fetch image data from the URL
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Convert the image data to a NumPy array
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        # Decode the image using OpenCV
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # Return the image
        return img
    else:
        return None

def getFaceData(video_url:str,search_class:str):
    """
    Identifies people in a video and returns their data based on the provided training data.

    Parameters:
    - video_url (str): URL of the video to be processed.
    - train_url (str): URL of the training data in JSON format.
    - search_class (str): Class name to search for in the training data (optional).

    Returns:
    A dictionary containing the number of people identified and their data.

    Additional Notes:
    - The function requires the face_recognition and OpenCV libraries.
    - If video_url is not provided, an error message is returned.
    - If the video fails to open, an error message is returned.
    - If there is an error fetching the training data, an error message is returned.
    - The function skips frames based on the frames per second (fps) of the video.
    - The matching threshold for face recognition is set to 0.5.
    - The data includes the name, roll number, and class of each identified person.
    - The function releases the video capture.
    """

    # sending a message to users to let them know what to do with this API
    if video_url == "":
        return {'message':'This API identifies people in a video and returns their data based on the provided training data.', 'Parameters':{"video_url (str)": "URL of the video to be processed.",'train_url (str)': 'URL of the training data in JSON format.','search_class (str)': 'Class name to search for in the training data (optional).'},'Returns':'A JSON containing the number of people identified and their data.'}

    # Open the video capture
    video_capture = cv2.VideoCapture(video_url)
    if not video_capture.isOpened():
         return {'message':'unable to open video, please try again with a valid video url'}

    train_url = "https://attendityprivateapi.onrender.com/display"
    params = {'class_name' : search_class.upper()}
    try:
        response = requests.get(train_url,params = params)
        response.raise_for_status()  # Check if the request was successful
        data_list = json.loads(response.json())
    except:
        return{'message':"Error while fetching data"}

    train_images = []
    classNames = []
    rollNumbers = []
    classes = []
    for data in data_list:
        curImg = read_image_from_url(data['imagePath'])
        print(data['imagePath'])
        if curImg is not None:
            train_images.append(curImg)
            classNames.append(data['name'])
            rollNumbers.append(data['rollNO'])
            classes.append(data['class_name'])

    encodeListKnown = []
    for img in train_images:
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(img)
            if len(face_encodings) > 0:
                encodeListKnown.append(face_encodings[0])

    # Get the total number of frames and calculate the frames per second (fps)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Calculate the frames to skip per second
    frames_to_skip = int(round(fps / 1))  # Process 1 frame per second

    # Initialize an empty list to store the attendance data
    attendance_data = []
    processed_rollnos = set()  # Set to store unique rollno

    frame_count = 0

    # Define the matching threshold
    matching_threshold = 0.7

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames if necessary
        if frame_count % frames_to_skip != 0:
            continue

        input_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(input_image_rgb, model="cnn")

        encodings = face_recognition.face_encodings(input_image_rgb, face_locations)

        for face_encoding in encodings:
            matches = face_recognition.compare_faces(encodeListKnown[:len(classNames)], face_encoding, tolerance=matching_threshold)
            face_distances = face_recognition.face_distance(encodeListKnown[:len(classNames)], face_encoding)

            for index, match in enumerate(matches):
              if match:
                name = classNames[index]
                roll_number = rollNumbers[index]
                class_name = classes[index]
                if roll_number in processed_rollnos:
                    continue  # Skip writing the record if rollno has already been processed
                print(name)
                processed_rollnos.add(roll_number)  # Add the roll number to the set of processed_rollnos
                attendance_data.append([name, roll_number, class_name])
    # Release the video capture
    video_capture.release()

    #prepairing data to return
    people_data = []
    people_identified = len(attendance_data)
    if len(attendance_data) > 0:
            for data in attendance_data:
                name, roll_number, class_name = data
                result = {
                    'name': name,
                    'roll_number': roll_number,
                    'class': class_name
                }
                people_data.append(result)

    return { "people_identified":people_identified,"data": people_data}

def convert_url(txt):
    if txt[-4]=='h':
	    txt=txt[-11::-1]
    else:
        txt=txt[::-1]
    x = txt.replace("/", "F2%",2 )
    x=x[::-1]
    return x

app = FastAPI()
@app.get("/")
def get_video(video_url: str = '',search_class: str = ''):
    video_url=convert_url(video_url)
    print(video_url)
    data = getFaceData(video_url,search_class)
    json_str = json.dumps(data, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

nest_asyncio.apply()
uvicorn.run(app,host='0.0.0.0', port=8000)

#sF7QMWz4_6hBHqPNzbBcgVWccWxiusiyqpFbFywoN
