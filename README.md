# Telugu Sign Language Recognition

This project aims to recognize Telugu sign language gestures using machine learning and Python to facilitate communication for the hearing and speech impaired. By focusing on regional language support, it addresses the lack of existing solutions that often require interpreters to translate into national languages. This project is designed to assist Telugu speakers who are hearing or speech impaired, providing them with a more direct and efficient means of communication.


## Installation Requirements

- Python
- OpenCV 
- MediaPipe 
- Scikit-learn 
- Matplotlib

    
## Implementation
### collect.py
Initially running this script collects images of hand gestures for different Telugu alphabets to build a dataset for training a classification model. Upon execution, it activates the webcam, guides the user to make specific gestures, and captures images in real-time. The images are organized into 52 folders, each representing a distinct alphabet, with 100 images per folder, forming the foundation for data processing and model training.

### create.py
This script processes collected images to extract hand gesture landmarks, creating a structured dataset for model training. Using the Mediapipe library, it employs a hand-tracking algorithm to identify and track anatomical points like fingertips and knuckles in real-time. The landmark data is then organized and labeled by alphabet, forming a crucial dataset for training the classification model.

### train.py
This script trains a machine learning model using a Random Forest Classifier to link hand landmarks with alphabet labels. It starts by loading the preprocessed dataset of hand landmarks and their labels, ensuring uniformity by adjusting the landmarks to a fixed length. The Random Forest Classifier, implemented with Scikit-learn, is then trained through multiple iterations to accurately predict alphabets based on hand gestures.

### inference.py
The Inference Classifier script is pivotal for real-time predictions of sign language gestures using the trained model. It captures video frames from the webcam, extracts hand landmarks, and feeds them into the classifier. The model then analyzes these inputs and identifies the corresponding alphabet, displaying the result on the video feed. This real-time inference enables practical application of the telugu sign language recognition system.







## Conclusion
In conclusion this project is particularly beneficial for people with hearing and speech impairments. By recognizing Telugu sign language gestures, it helps them communicate more effectively with their peers in regional language.It reduces the reliance on interpreters, enabling people to interact independently and confidently. By focusing on Telugu, this project respects and promotes regional language use, which is crucial in preserving cultural identity and ensuring that people  feel valued and understood in their native language.
