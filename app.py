from flask import Flask, render_template, Response, jsonify
import cv2 as cv
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)

model=YOLO('best.pt')

video = cv.VideoCapture('cr.mp4')

def generate_frames():
    box_entry_time = {}

    time_account_by_id = {}
    while True:
        success, frame = video.read()
        
        if not success:
            video.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model.track(frame, persist=True)
        # class_names = model.names
        
        qtd_atendentes = 0
        qtd_clientes = 0
        for result in results:
            for i in range(len(result.boxes)):
                class_name = result.names[result.boxes[i].cls[0].item()]
                if class_name.strip() == 'Cliente':
                    qtd_clientes += 1
                    # print(class_name)
                    box_id = int(result.boxes[i].id)
                    list_de_box_ids = [4, 5]
                    dicionario = {1, 4, 5}

                    # Rotina para gerar informações sobre o tempo de atendimento do cliente.
                    current_time = datetime.now()
                    if box_id not in time_account_by_id:
                        time_account_by_id[box_id] = [
                            [current_time.hour, current_time.minute, current_time.second, current_time.microsecond // 1000,current_time.day, current_time.month, current_time.year], 
                            [current_time.hour, current_time.minute, current_time.second, current_time.microsecond // 1000, current_time.day, current_time.month, current_time.year],
                            [current_time.hour, current_time.minute, current_time.second, current_time.microsecond // 1000, current_time.day, current_time.month, current_time.year],  
                            0
                            ]
                    else:
                        time_account_by_id[box_id][2] = [current_time.hour, current_time.minute, current_time.second, current_time.microsecond // 1000, current_time.day, current_time.month, current_time.year]

                        diff_seconds = (
                            datetime(current_time.year, current_time.month, current_time.day,
                                    time_account_by_id[box_id][2][0], time_account_by_id[box_id][2][1], time_account_by_id[box_id][2][2],
                                    time_account_by_id[box_id][2][3])
                            - datetime(current_time.year, current_time.month, current_time.day,
                                        time_account_by_id[box_id][0][0], time_account_by_id[box_id][0][1], time_account_by_id[box_id][0][2],
                                        time_account_by_id[box_id][0][3])
                        ).total_seconds()

                        hours, remainder = divmod(diff_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        seconds, milliseconds = divmod(int(seconds * 1000), 1000)

                        # time_account_by_id[box_id][3] = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                        time_account_by_id[box_id][3] = f"{int(hours)}h {int(minutes)}m {int(seconds)}s {milliseconds}ms"
                    
                        if time_account_by_id[box_id][2] != time_account_by_id[box_id][1]:
                            print('hahahaha')
                            print(box_id)
                            print(time_account_by_id[box_id][0])
                            print(time_account_by_id[box_id][1])
                            print(time_account_by_id[box_id][2])
                            print(time_account_by_id[box_id][3])
                            print('hahahaha')
                            time_account_by_id[box_id][1] = time_account_by_id[box_id][2]
                        elif time_account_by_id[box_id][2] == time_account_by_id[box_id][1]:
                            # escrever aqui a rotina de CSV
                            # time_account_by_id.pop([box_id])
                            print('hihihihi')


                    # Rotina para plotar as caixas e o tempo decorrido
                    box_coordinates = result.boxes[i].xyxy[0].cpu().numpy()
                    # Verifique se o ID já está no dicionário
                    if box_id not in box_entry_time:
                        # Se não estiver, adicione o ID ao dicionário com o tempo atual
                        box_entry_time[box_id] = datetime.now()
                    # Calcule o tempo de permanência
                    entry_time = box_entry_time[box_id]
                    current_time = datetime.now()
                    time_difference = current_time - entry_time
                    seconds_passed = time_difference.total_seconds()
                    # Desenhe o tempo no frame
                    cv.putText(
                        frame,
                        # f"ID {box_id}: {int(seconds_passed)}s",
                        f"Tempo de espera: {int(seconds_passed)}s",
                        (int(box_coordinates[0]), int(box_coordinates[1] - 30)),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                else:
                    qtd_atendentes += 1

            print(time_account_by_id)
            frame_predicted = result.plot(boxes=True, conf=False) # Track

            _, buffer = cv.imencode('.jpg', frame_predicted)
            frame_final = buffer.tobytes()
            app.config['qtd_atendentes'] = qtd_atendentes
            app.config['qtd_clients'] = qtd_clientes
            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_final + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_counts')
def get_counts():
    qtd_atendentes = app.config['qtd_atendentes']
    qtd_clientes = app.config['qtd_clients']
    return jsonify(
        {
            'qtd_atendentes': qtd_atendentes,
            'qtd_clientes': qtd_clientes
        }
    )

if __name__ == '__main__':
    app.run(debug=True)