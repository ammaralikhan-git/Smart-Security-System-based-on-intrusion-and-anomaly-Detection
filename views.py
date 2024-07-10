from roboflow import Roboflow
import supervision as sv
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import logout
from .forms import LoginForm
from .models import CusUser
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.http import JsonResponse
import os
import cv2
from .models import FaceImage  # Import the FaceImage model
from django.core.files.base import ContentFile
from PIL import Image
from io import BytesIO
import base64
from django.utils import timezone
import random
import string
from django.conf import settings
from django.core.files.base import ContentFile
from facedetection.models import FaceImage
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import base64
import json
from .recognizer import recognize_faces
import numpy as np
import face_recognition
import pickle
import io
import traceback


# Load known face encodings and names
with open('facedetection/src/face_encodings.pkl', 'rb') as file:
    encodings_dict = pickle.load(file)

known_encodings = []
known_names = []

for name, encodings in encodings_dict.items():
    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise Exception("Failed to load Haar Cascade classifier.")

def recognize_faces(frame):
    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    results = []

    if not face_locations:
        print("No faces detected.")
    else:
        print(f"{len(face_locations)} faces detected.")

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        name = "Unknown"
        if matches[best_match_index]:
            name = known_names[best_match_index]

        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        result = {
            'name': name,
            'bbox': [left, top, right - left, bottom - top],  # Convert coordinates to int
            'confidence': float(face_distances[best_match_index]) if matches[best_match_index] else None
        }
        results.append(result)
    return results

@csrf_exempt
def start_face_detection(request):
    if request.method == "POST":
        data = json.loads(request.body)
        image_data = data.get('image', '')
        if not image_data:
            return JsonResponse({'status': 'error', 'message': 'No image data provided'}, status=400)

        try:
            format, imgstr = image_data.split(';base64,')
            image = np.array(Image.open(io.BytesIO(base64.b64decode(imgstr))))
            recognized_faces = recognize_faces(image)

            return JsonResponse({
                'status': 'success',
                'detections': recognized_faces
            })
        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            print(error_message)
            return JsonResponse({'status': 'error', 'message': error_message}, status=500)
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)



# # Initialize the model on startup
# def load_model():
#     weights_path = 'facedetection/src/yolov4-obj_final.weights'
#     config_path = 'facedetection/src/yolov4-obj.cfg'
#     names_path = 'facedetection/src/obj.names'

#     net = cv2.dnn.readNet(weights_path, config_path)
#     with open(names_path, "r") as f:
#         classes = [line.strip() for line in f.readlines()]
#     return net, classes

# net, classes = load_model()

# def detect_weapons(img, net, classes):
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#     height, width, _ = img.shape
#     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#     print("outs value:", outs)
#     print("outs type", type(outs))
#     class_ids = []
#     confidences = []
#     boxes = []
#     results = []
#     has_detection = False
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             #print("Confidence: {:.2f}".format(confidence))  # Print confidence value
            
            
#             if confidence > 0.90:
#                 print("conf values in threshold", confidence)
#                 has_detection = True
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             confidence = confidences[i]
            
#             results.append({
#             'type': label,
#             'confidence': confidence,
#             'bbox': [x, y, w, h],
#         })

#     return results
# Initialize the Roboflow model on startup
def load_model():
    rf = Roboflow(api_key="nrl9YUpgRfJhzvvrwn2K")
    project = rf.workspace().project("weapons-yolo")
    model = project.version(2).model
    return model

model = load_model()

def detect_and_annotate(img, model):
    result = model.predict(img, confidence=90, overlap=50).json()  # Increased confidence and overlap thresholds
    print("Predictions:", result["predictions"])  # Log predictions for debugging

    if not result["predictions"]:
        print("No weapons detected.")  # Explicitly log when no weapons are detected

    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_roboflow(result)
    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()
    annotated_img = bounding_box_annotator.annotate(scene=img, detections=detections)
    annotated_img = label_annotator.annotate(scene=annotated_img, detections=detections, labels=labels)
    
    # Check for handgun or rifle
    weapon_detected = any(label in ["handgun", "rifle"] for label in labels)

    return annotated_img, weapon_detected

@csrf_exempt
def start_weapon_detection(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)
    try:
        data = json.loads(request.body)
        image_data = data.get('image', '')
        if not image_data:
            return JsonResponse({'status': 'error', 'message': 'No image data provided'}, status=400)

        format, imgstr = image_data.split(';base64,')
        img_decoded = base64.b64decode(imgstr)
        img = cv2.imdecode(np.frombuffer(img_decoded, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return JsonResponse({'status': 'error', 'message': 'Failed to decode the image'}, status=400)

        annotated_img, weapon_detected = detect_and_annotate(img, model)
        
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_img_str = base64.b64encode(buffer).decode('utf-8')
        
        return JsonResponse({'status': 'success', 'annotated_image': f'data:image/jpeg;base64,{annotated_img_str}', 'weapon_detected': weapon_detected})
    
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
# @csrf_exempt
# def start_weapon_detection(request):
#     if request.method != 'POST':
#         return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

#     try:
#         data = json.loads(request.body)
#         image_data = data.get('image', '')
#         if not image_data:
#             return JsonResponse({'status': 'error', 'message': 'No image data provided'}, status=400)

#         format, imgstr = image_data.split(';base64,')
#         img_decoded = base64.b64decode(imgstr)
#         img = cv2.imdecode(np.frombuffer(img_decoded, np.uint8), cv2.IMREAD_COLOR)

#         if img is None:
#             return JsonResponse({'status': 'error', 'message': 'Failed to decode the image'}, status=400)

#         # Debugging: Save the image to disk
#         #cv2.imwrite("decodeedimage", img)
#         print("Image Shape:", img.shape)

#         detected_weapons = detect_weapons(img, net, classes)
#         return JsonResponse({'status': 'success', 'weapons': detected_weapons})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
def logout_view(request):
    logout(request)
    return redirect('index')  # Redirect to the index page or any other page you want


def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            try:
                user = CusUser.objects.get(username=username)
                if user.password == password:
                    # Authentication successful
                    # Log in the user
                    request.session['user_id'] = user.id
                    # Redirect to a success page or any other desired page
                    return redirect('main')  # Replace 'main' with the name of your main page URL
                else:
                    # Handle invalid password
                    form.add_error('password', 'Invalid password')
            except CusUser.DoesNotExist:
                # Handle user not found
                form.add_error('username', 'User not found')
    else:
        form = LoginForm()
    return render(request, 'facedetection/login.html', {'form': form})



def register(request):
    return render(request, 'facedetection/registration_page1.html')


def main(request):
    try:
        # Retrieve all face images from the database
        face_images = FaceImage.objects.all()

        # Pass the face images to the template context
        context = {'face_images': face_images}

        # Render the template with the context
        return render(request, 'facedetection/main.html', context)
    except Exception as e:
        print("Error in main view:", str(e))


def index(request):
    return render(request, 'facedetection/login.html')

def registration_page1(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        surname = request.POST.get('surname')
        username = request.POST.get('username')
        # You can handle form validation here if needed
        # Save data to session
        request.session['page1_data'] = {
            'first_name': first_name,
            'surname': surname,
            'username': username
        }
        return redirect('registration_page2')
    return render(request, 'facedetection/registration_page1.html')

def registration_page2(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        # You can handle form validation here if needed
        # Save data to session
        request.session['page2_data'] = {
            'email': email,
            'phone': phone
        }
        return redirect('registration_page3')
    return render(request, 'facedetection/registration_page2.html')

def registration_page3(request):
    if request.method == 'POST':
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        # You can handle form validation here if needed
        # Save data to session
        request.session['page3_data'] = {
            'password': password,
            'confirm_password': confirm_password
        }
        # Combine all data from previous pages
        user_data = {
            **request.session['page1_data'],
            **request.session['page2_data'],
            **request.session['page3_data']
        }
        # Remove confirm_password from user_data
        user_data.pop('confirm_password', None)
        # Save user_data to the database
        new_user = CusUser.objects.create(**user_data)
        # Optionally, you can perform additional actions after saving the user, such as sending a confirmation email
        # ...
        return redirect('login')
    return render(request, 'facedetection/registration_page3.html')




# Get the base directory of your Django project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the Haar cascade XML file
CASCADE_XML_PATH = os.path.join(BASE_DIR, 'data', 'haarcascades/haarcascade_frontalface_default.xml')

# Initialize the cascade classifier with the correct path
face_cascade = cv2.CascadeClassifier(CASCADE_XML_PATH)



def generate_random_room_number():
    """Generate a random room number between 1 and 4."""
    return str(random.randint(1, 4))
    
def save_face_to_database(username, base64_image):
    try:
        # Convert base64 string to binary data
        image_data = base64.b64decode(base64_image)

        # Get the path to the media directory
        media_path = os.path.join(settings.BASE_DIR, 'media')
        face_images_path = os.path.join(media_path, 'face_images')

        # Ensure that the directory exists, if not, create it
        if not os.path.exists(face_images_path):
            os.makedirs(face_images_path)

        # Create a unique filename for the image
        filename = os.path.join(face_images_path, f'{username}_face.jpg')

        # Save the image to the media directory
        with open(filename, 'wb') as f:
            f.write(image_data)

        # Save FaceImage object to the database
        face_image = FaceImage.objects.create(
            username=username,
            image_data=f'face_images/{username}_face.jpg',
            room_number=generate_random_room_number(),
            timestamp=timezone.now()
        )
        face_image.save()
    except Exception as e:
        print("Error saving face to database:", str(e))