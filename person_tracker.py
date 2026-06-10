from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(4, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    persons = []

    for box in results[0].boxes:

        cls = int(box.cls[0])

        if cls == 0:  # person

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            area = (x2 - x1) * (y2 - y1)

            persons.append((area, x1, y1, x2, y2))

    if len(persons) > 0:

        persons.sort(reverse=True)

        area, x1, y1, x2, y2 = persons[0]

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"X:{center_x}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("AI Fan Tracker", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()