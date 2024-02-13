import cv2 as cv
from ultralytics import YOLO
from datetime import datetime

# Instancie o seu modelo YOLOv8 e carregue os pesos treinados
yolo_model = YOLO('best_80.pt')

video = cv.VideoCapture('video_2.mp4')

box_entry_time = {}
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Realize a detecção de objetos no frame
    results = yolo_model.track(frame, persist=True)

    # Acesse as informações de cada detecção (box, confiança, classe)
    for result in results:
        for i in range(len(result.boxes)):
            box_id = int(result.boxes[i].id)

            coordenadas = result.boxes[i].xyxyn
            x_min_norm, y_min_norm, x_max_norm, y_max_norm = coordenadas[0]
            height, width = frame.shape[:2]

            # Coordenadas absolutas
            x_min_abs = int(x_min_norm * width)
            y_min_abs = int(y_min_norm * height)
            x_max_abs = int(x_max_norm * width)
            y_max_abs = int(y_max_norm * height)

            classe = result.names[result.boxes[i].cls[0].item()]

            if classe.strip() == 'Cliente':
                if box_id not in box_entry_time:
                        # Se não estiver, adicione o ID ao dicionário com o tempo atual
                        box_entry_time[box_id] = datetime.now()
                # Calcule o tempo de permanência
                entry_time = box_entry_time[box_id]
                current_time = datetime.now()
                time_difference = current_time - entry_time
                seconds_passed = time_difference.total_seconds()
                # Desenhe o tempo no frame


                if seconds_passed > 15:
                    cv.rectangle(frame, (x_min_abs, y_min_abs), (x_max_abs, y_max_abs), (0, 255, 0), 2)
                    cv.putText(
                        frame,
                        # f"ID {box_id}: {int(seconds_passed)}s",
                        f"ID: {box_id} em espera: {int(seconds_passed)}s",
                        (x_min_abs, y_min_abs-3),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
            elif classe.strip() == 'Atendente':
                ...
                # cv.rectangle(frame, (x_min_abs, y_min_abs), (x_max_abs, y_max_abs), (0, 0, 255), 2)

    # frame_predicted = result.plot(boxes=True, conf=False) # Track
    cv.imshow('Frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos
video.release()
cv.destroyAllWindows()
