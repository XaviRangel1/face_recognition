from face_detection import FaceDetection
from face_recognition import FaceRecognition
import cv2

if __name__ == "__main__":
    # Init the Face Detection and Face Recognition classes
    detection = FaceDetection()
    recognition = FaceRecognition()

    # Read the image
    image = cv2.imread('./yolov3/data/samples/person.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect a face in the image (if many, the biggest one)
    bounding_box = detection.detect(image)

    # bounding_box is a dictionary with parameters: x1, y1, x2, y2, width, height
    print(bounding_box)

    # Plot the bounding box on the image
    detection.plot(image, bounding_box)

    # Extract the face from the image
    face = recognition.extract(image, bounding_box)

    # Check if the face is from an employee, return True or False
    is_employee = recognition.recognize(face)

    if is_employee:
        print('Opening Door')


        