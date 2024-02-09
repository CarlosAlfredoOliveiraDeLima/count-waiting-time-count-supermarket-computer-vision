from flask import Flask, render_template, Response, jsonify
import cv2 as cv
from ultralytics import YOLO
from datetime import datetime
import os
import csv

app = Flask(__name__)

model=YOLO('best_30.pt')

video = cv.VideoCapture('video_1.mp4')

nome_arquivo_csv = "dados_tempo_atendimento.csv"
cabecalho = ["ID", "hora_entrada", "data_entrada", "hora_saida", "data_saida", "tempo_de_permanencia"]


def generate_frames():
    box_entry_time = {}

    time_account_by_id = {}
    while True:
        success, frame = video.read()
        
        if not success:
            video.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model.track(frame, persist=True, verbose=False, show=False)
        # class_names = model.names
        

        qtd_atendentes = 0
        qtd_clientes = 0
        for result in results:
            IDs_atuais = []
            for i in range(len(result.boxes)):
                if result.boxes[i].id is None:
                    _, buffer = cv.imencode('.jpg', frame)
                    frame_final = buffer.tobytes()
                    app.config['qtd_atendentes'] = qtd_atendentes
                    app.config['qtd_clients'] = qtd_clientes
                    yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame_final + b'\r\n')
                    continue


                class_name = result.names[result.boxes[i].cls[0].item()]
                if class_name.strip() == 'Cliente':
                    qtd_clientes += 1
                    box_id = int(result.boxes[i].id)
                    IDs_atuais.append(box_id)

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
                            time_account_by_id[box_id][1] = time_account_by_id[box_id][2]


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

            IDs_para_serem_deletados_do_dicionario = []
            for chave_id in time_account_by_id.keys():
                if chave_id not in IDs_atuais:
                    print(time_account_by_id[chave_id]) #Fazer um log
                    IDs_para_serem_deletados_do_dicionario.append(chave_id)
                    csv_existe = os.path.isfile(nome_arquivo_csv)

                    with open(nome_arquivo_csv, mode='a', newline='') as arquivo_csv:
                        escritor_csv = csv.writer(arquivo_csv)

                        if not csv_existe:
                            escritor_csv.writerow(cabecalho)

                        id_csv = chave_id
                        hora_entrada_csv = f'{time_account_by_id[chave_id][0][0]}:{time_account_by_id[chave_id][0][1]}:{time_account_by_id[chave_id][0][2]}'
                        data_entrada_csv = f'{time_account_by_id[chave_id][0][4]}/{time_account_by_id[chave_id][0][5]}/{time_account_by_id[chave_id][0][6]}'
                        hora_saida_csv = f'{time_account_by_id[chave_id][2][0]}:{time_account_by_id[chave_id][2][1]}:{time_account_by_id[chave_id][2][2]}'
                        data_saida_csv = f'{time_account_by_id[chave_id][2][4]}/{time_account_by_id[chave_id][2][5]}/{time_account_by_id[chave_id][2][6]}'
                        tempo_permanencia_csv = time_account_by_id[chave_id][3]
                        escritor_csv.writerow([id_csv, hora_entrada_csv, data_entrada_csv, hora_saida_csv, data_saida_csv, tempo_permanencia_csv])

            for id_del_dict in IDs_para_serem_deletados_do_dicionario:    
                del time_account_by_id[id_del_dict]

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
    app.run(debug=False)