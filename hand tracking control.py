import cv2
import mediapipe as mp
import time
from pycaw.pycaw import AudioUtilities

# กำหนดค่าสำหรับ hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# เริ่มต้นเครื่องมือตรวจจับมือ
with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
    # เริ่มต้นกล้อง
    cap = cv2.VideoCapture(0)
    
    # ตรวจสอบอุปกรณ์เสียง
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        AudioUtilities.IAudioEndpointVolume._iid_, 
        CLSCTX_ALL, 
        None
    )
    
    volume = interface.GetMasterVolumeLevel()
    
    while cap.isOpened():
        # อ่านภาพจากกล้อง
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # เปลี่ยนสีภาพจาก BGR เป็น RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # ใช้ Mediapipe เพื่อค้นหามือ
        results = hands.process(image)

        # วาดเส้นติดตามมือและหัวไหล่
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ดึงค่าพิกัดของนิ้วชี้และนิ้วโป้ง
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # คำนวณระยะห่างระหว่างนิ้วชี้และนิ้วโป้ง
                distance = abs(index_finger.x - thumb.x)

                # แสดงจุดบนหน้าจอเมื่อนิ้วชี้กำลังชี้
                if distance < 0.05:
                    x, y = int(index_finger.x * image.shape[1]), int(index_finger.y * image.shape[0])
                    cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

                    # ปรับเสียงตามการขยับนิ้ว
                    volume_delta = int((x - image.shape[1]//2) / image.shape[1] * 10)
                    volume += volume_delta
                    volume = max(-65, min(volume, 0))
                    interface.SetMasterVolumeLevel(volume, None)
                    
        # แสดงภาพที่ถูกประมวลผล
        cv2.imshow('Hand Tracking', image)

        # หยุดการทำงานเมื่อกด 'q' บนแป้นพิมพ์
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # คืนทรัพยากร
    cap.release()
    cv2.destroyAllWindows()
