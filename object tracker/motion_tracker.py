# All these libraries are nesecary for the code to run and function make sure they are installed before running 
# or else the code will not run. 
# The code currently tracks objects and is able to identify them, though I noticed it had troubles with objects
# that were in the background and not recognizing them.

import cv2                                          # OpenCV library for video capture, this take and processes all the live camera
import mediapipe as mp                              # Google's MediaPipe framework for machine learning tasks
from mediapipe.tasks import python                  # MediaPipe Python core tasks interface
from mediapipe.tasks.python import vision           # MediaPipe vision tasks handles all the tracking and machine learning 


# points to our model file
model_path = 'efficientdet_lite0.tflite'

# sets up the base options using that model file path
base_options = python.BaseOptions(model_asset_path=model_path)

# sets up detection rules like needing at least 50% confidence and capping at 5 results
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,    # Ignore any detected objects with less than 50% confidence
    max_results=5           # Limit the maximum number of simultaneous objects tracked per frame
)

# builds the actual object detector using the options we just set
detector = vision.ObjectDetector.create_from_options(options)

# starts up the webcam 0 is there since I only have one camera but if there 
#were multiple cameras and wanted to use a different one then we'd could change that.
cap = cv2.VideoCapture(0)

# makes sure the webcam is opened
if not cap.isOpened():
    print("Webcam was not opened.")
    exit()

print("Starting Object Detection... Press 'q' to quit.")

while cap.isOpened():
    # reads the camera frame and checks if anything was grabbed
    success, frame = cap.read()
    if not success:
        print("No image was grabbed.")
        continue

    # flips the video horizontally 
    frame = cv2.flip(frame, 1)
    
    # gets the height and width of the frame for calculations later
    h, w, _ = frame.shape

    # This sections converts BGR to RBG since thats necessary for mediapipe 
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # wraps the frame into a special image format that mediapipe can read
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)


    # runs the detector on our image to find objects, boxes, and confidence scores
    detection_result = detector.detect(mp_image)

    # This section loops through every object the model picked up in this frame
    for detection in detection_result.detections:
        # grabs the box data for the detected object
        bbox = detection.bounding_box
        
        # sets the edges of the box based on the coordinates
        start_x = bbox.origin_x                     # Left edge of the object box
        start_y = bbox.origin_y                     # Top edge of the object box
        end_x = bbox.origin_x + bbox.width          # Right edge of the object box
        end_y = bbox.origin_y + bbox.height         # Bottom edge of the object box

        # finds the exact horizontal middle of the object
        obj_center_x = start_x + (bbox.width // 2)

        # checks which part of the screen the object is sitting in
        if obj_center_x < w / 3:
            zone = "Left Zone"      # Object is on the left side of the user's view
        elif obj_center_x < 2 * (w / 3):
            zone = "Center Zone"    # Object is directly ahead in the user's path
        else:
            zone = "Right Zone"     # Object is on the right side of the user's view

        # grabs what the object is and turns its score into a percentage
        category = detection.categories[0]       # Get the primary predicted category
        label_name = category.category_name      # Name of the object (e.g., 'chair', 'person', 'bottle')
        score = round(category.score * 100, 1)   # Convert confidence decimal into a readable percentage


        # This protion creates a "frame" around the object that is detected 
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Format the text display string (e.g., "chair (Center Zone) 85.4%")
        display_text = f"{label_name} ({zone}) {score}%"
        
        # Calculate text position (place just above the bounding box, or slightly below if too close to top)
        y_text = start_y - 15 if start_y - 15 > 15 else start_y + 15
        
        # Render the text overlay onto the video frame
        cv2.putText(frame, display_text, (start_x, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    # Renders the processed frame into an interactive GUI preview window
    cv2.imshow('NeuroLens - MediaPipe Object Detection', frame)

    # Closes the program once "q" is hit on the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()               # Turns off the camera 
cv2.destroyAllWindows()     # Closes active OpenCV windows