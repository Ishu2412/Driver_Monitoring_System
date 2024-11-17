import cv2
import supervision as sv
from ultralytics import YOLO
from supervision import BoxAnnotator
import threading
import time
import numpy as np
import dlib
from scipy.spatial import distance as dist


model = YOLO("best.pt")
box_annotator = BoxAnnotator()  

drowsiness_score = 0 
DROWSINESS_THRESHOLD = 5  
YAWN_THRESH = 30  
alarm_status2 = False


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance


def play_beep():
    """Plays a beep sound in a separate thread."""
    winsound.Beep(1000, 5000)  


def main():
    global drowsiness_score, alarm_status2

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera is not opened")
        return

    while cam.isOpened():
        success, frame = cam.read()
        if success:
            
            predictions = model(frame)[0]
            detections = sv.Detections.from_ultralytics(predictions)

            
            annotated_frame = box_annotator.annotate(scene=frame, detections=detections)

           
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = np.array([[p.x, p.y] for p in shape.parts()])

                
                distance = lip_distance(shape)

               
                lip = shape[48:60]
                cv2.drawContours(annotated_frame, [lip], -1, (0, 255, 0), 1)

                
                if distance > YAWN_THRESH:
                    cv2.putText(annotated_frame, "Yawn Alert", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not alarm_status2:
                        alarm_status2 = True
                        threading.Thread(target=play_beep).start()  
                else:
                    alarm_status2 = False

            
            detected_labels = []
            for i, (box, confidence, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
                label = f"{model.model.names[class_id]}: {confidence:.2f}"
                detected_labels.append(model.model.names[class_id])
                (x_min, y_min, x_max, y_max) = map(int, box)

                
                cv2.putText(annotated_frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            
            if 'Closed Eye' in detected_labels:
                drowsiness_score += 1
            elif 'Open Eye' in detected_labels:
                drowsiness_score = max(0, drowsiness_score - 1)  

            
            if drowsiness_score >= DROWSINESS_THRESHOLD:
                cv2.putText(annotated_frame, "DROWSINESS DETECTED", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                threading.Thread(target=play_beep).start() 
                drowsiness_score = 0

            
            labels_to_check = ['Open Eye', 'Closed Eye', 'Cigarette', 'Phone', 'Seatbelt']
            for idx, label in enumerate(labels_to_check):
                status = "Detected" if label in detected_labels else "Not Detected"
                cv2.putText(annotated_frame, f"{label}: {status}", (10, 100 + idx * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if status == "Detected" else (0, 0, 255), 2)

           
            cv2.imshow("Detection", annotated_frame)

            
            if cv2.waitKey(1) == ord("q"):
                break

        else:
            raise Exception("Can't Open Camera")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
