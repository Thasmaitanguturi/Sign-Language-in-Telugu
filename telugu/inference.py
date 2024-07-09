import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Specify the path to the Telugu font
font_path = r"C:\Users\Thasmai\Downloads\Pothana2000.ttf"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'అ', 1: 'ఆ', 2: 'ఇ', 3: 'ఈ', 4: 'ఉ', 5: 'ఊ', 6: 'ఋ', 7: 'ౠ', 8: 'ఎ', 9: 'ఏ',
               10: 'ఐ', 11: 'ఒ', 12: 'ఓ', 13: 'ఔ', 14: 'క', 15: 'ఖ', 16: 'గ', 17: 'ఘ', 18: 'ఙ', 19: 'చ',
               20: 'ఛ', 21: 'జ', 22: 'ఝ', 23: 'ఞ', 24: 'ట', 25: 'ఠ', 26: 'డ', 27: 'ఢ', 28: 'ణ', 29: 'త',
               30: 'థ', 31: 'ద', 32: 'ధ', 33: 'న', 34: 'ప', 35: 'ఫ', 36: 'బ', 37: 'భ', 38: 'మ', 39: 'య',
               40: 'ర', 41: 'ల', 42: 'వ', 43: 'శ', 44: 'ష', 45: 'స', 46: 'హ', 47: 'ళ', 48: 'క్ష', 49: 'ఱ',
               50: 'అం', 51: 'అః'}

# Load the Telugu font
telugu_font = ImageFont.truetype(font_path, 32)

while True:
    try:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        if not ret:
            raise Exception("Error: Unable to read frame from the camera.")

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                # Use Pillow to draw text on the image
                pil_image = Image.fromarray(frame)
                draw = ImageDraw.Draw(pil_image)
                draw.text((x1, y1 - 10), predicted_character, font=telugu_font, fill=(0, 0, 0))
                frame = np.array(pil_image)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    except Exception as e:
        print(e)

cap.release()
cv2.destroyAllWindows()






