import cv2 as cv
from ultralytics import YOLO
from datetime import datetime

# Instancie o seu modelo YOLOv8 e carregue os pesos treinados
yolo_model = YOLO('best_80.pt')

video = cv.VideoCapture('video_2.mp4')

box_entry_time_waiting = {}
box_entry_time_service = {}
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Realize a detecção de objetos no frame
    results = yolo_model.track(frame, persist=True)

    frame = cv.rectangle(frame, (530,180), (750,300), (255,255,0), 2)

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
                # if box_id not in box_entry_time_waiting:
                #     # Se não estiver, adicione o ID ao dicionário com o tempo atual
                #     box_entry_time_waiting[box_id] = datetime.now()
                # # Calcule o tempo de permanência
                # entry_time = box_entry_time_waiting[box_id]
                # current_time = datetime.now()
                # time_difference = current_time - entry_time
                # seconds_passed = time_difference.total_seconds()



                # if seconds_passed > 15:
                #     cv.rectangle(frame, (x_min_abs, y_min_abs), (x_max_abs, y_max_abs), (0, 255, 0), 2)
                #     cv.putText(
                #         frame,
                #         # f"ID {box_id}: {int(seconds_passed)}s",
                #         f"ID: {box_id} ESPERA: {int(seconds_passed)}s",
                #         (x_min_abs, y_min_abs-3),
                #         cv.FONT_HERSHEY_SIMPLEX,
                #         0.5,
                #         (255, 255, 255),
                #         2,
                #     )

                centroide_x = (x_min_abs + x_max_abs) // 2
                centroide_y = (y_min_abs + y_max_abs) // 2
                coordenadas_centroide = (centroide_x, centroide_y)
                frame = cv.circle(frame, coordenadas_centroide, 10, (255, 255, 224), thickness=cv.FILLED)


                if (530 <=  centroide_x <= 750) and (180 <= centroide_y <= 300):
                    if box_id not in box_entry_time_service:
                    # Se não estiver, adicione o ID ao dicionário com o tempo atual
                        box_entry_time_service[box_id] = datetime.now()
                    # Calcule o tempo de permanência
                    entry_service_time = box_entry_time_service[box_id]
                    current_time = datetime.now()
                    time_difference_service = current_time - entry_service_time
                    seconds_passed_service = time_difference_service.total_seconds()


                    cv.putText(
                        frame,
                        # f"ID {box_id}: {int(seconds_passed)}s",
                        f"ID: {box_id} SERVICO: {int(seconds_passed_service)}s",
                        (x_min_abs, y_min_abs-3),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                else:
                    if box_id not in box_entry_time_waiting:
                        # Se não estiver, adicione o ID ao dicionário com o tempo atual
                        box_entry_time_waiting[box_id] = datetime.now()
                    # Calcule o tempo de permanência
                    entry_time = box_entry_time_waiting[box_id]
                    current_time = datetime.now()
                    time_difference = current_time - entry_time
                    seconds_passed = time_difference.total_seconds()



                    if seconds_passed > 15:
                        cv.rectangle(frame, (x_min_abs, y_min_abs), (x_max_abs, y_max_abs), (0, 255, 0), 2)
                        cv.putText(
                            frame,
                            # f"ID {box_id}: {int(seconds_passed)}s",
                            f"ID: {box_id} ESPERA: {int(seconds_passed)}s",
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
