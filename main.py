from face_detection import FaceDetection
from face_recognition import FaceRecognition
from utils import read_image

if __name__ == "__main__":
    # Init the Face Detection and Face Recognition classes
    detection = FaceDetection()
    recognition = FaceRecognition()

    # Read the image
    image = read_image('./yolov3/data/samples/person.jpg')

    # Detect a face in the image (if many, returns the biggest one; if none, returns None)
    bounding_box = detection.detect(image)

    # bounding_box is a dictionary with parameters: x1, y1, x2, y2, width, height
    print(bounding_box)

    if bounding_box is not None:
        # Plot the bounding box on the image
        detection.plot(image, bounding_box)

        # Extract the face from the image
        face = recognition.extract(image, bounding_box)

        # Check if the face is from an employee, return True or False
        is_employee = recognition.recognize(face)

        if is_employee:
            print('Opening Door')