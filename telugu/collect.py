import os
import cv2

DATA_DIR = './telugu_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


number_of_classes = 52
dataset_size = 100

# Attempt to open the camera
cap = cv2.VideoCapture(0)  # Use the default camera (index 0)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    print('Press "Q" when ready.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Exit if 'Q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()

