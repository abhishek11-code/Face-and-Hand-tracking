import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands, \
     mp_face.FaceDetection(min_detection_confidence=0.7) as face_detection:

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)  
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            face_results = face_detection.process(rgb_frame)
            if face_results.detections:
                for detection in face_results.detections:
                    mp_drawing.draw_detection(frame, detection)
                cv2.putText(frame, f'Faces: {len(face_results.detections)}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            
            hand_results = hands.process(rgb_frame)
            total_fingers = 0

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = hand_landmarks.landmark
                    fingers = []

                    
                    fingers.append(landmarks[mp_hands.HandLandmark.THUMB_TIP].x >
                                   landmarks[mp_hands.HandLandmark.THUMB_IP].x)

                   
                    for tip_id in [8, 12, 16, 20]:
                        fingers.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)

                    total_fingers += fingers.count(True)

            cv2.putText(frame, f'Fingers: {total_fingers}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)

            
            cv2.imshow("Face and Finger Detection", frame)

           
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested.")
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")

    finally:
        
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")
