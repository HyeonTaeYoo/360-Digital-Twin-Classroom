from flask import Flask, render_template, Response, request
from camera_360 import Camera360
import time
import json
import frame_face_grab
import math
import glob
import os
import test
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/virtualSpatial')
def virtual():
    path = '.\\static\\ppt\\*'
    file_list = glob.glob(path)
    imglist = [file for file in file_list if file.endswith(".png")]
    imglist.sort()
    return render_template('webGL.html', imglist=imglist)


@app.route('/realSpatial')
def real():
    return render_template('index.html')


@app.route('/Hair', methods=['GET'])
def index():
    # Main page
    return render_template('hair.html')


@app.route('/Avatar', methods=['GET'])
def avatar():
    return render_template('avatar.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # print(type(f))   # filestorage

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = test.model_predict(file_path)

        if preds == '61':
            preds = "The hairstyle is a ht hair" + " (" + preds + ") "
        elif preds == '56':
            preds = "The hairstyle is a jae cut" + " (" + preds + ") "
        elif preds == '114':
            preds = "The hairstyle is a ys cut" + " (" + preds + ") "
        elif preds == '102':
            preds = "The hairstyle is a joo and jin cut" + " (" + preds + ") "
        return preds
    return None


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feeding():
    return Response(gen(Camera360()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 두 직선의 교점을 구함


def get_intersect_point(a1, b1, c1, d1, a2, b2, c2, d2):
    x = (b2 - b1 + a1 * ((b1 - d1) / (a1 - c1)) - a2 *
         ((b2-d2) / (a2-c2))) / ((b1-d1)/(a1-c1) - (b2-d2) / (a2 - c2))
    y = ((b1 - d1) / (a1 - c1)) * (x - a1) + b1

    return x, y


def get_position_in_classroom(name_xy_coord):
    if len(name_xy_coord) == 5:  # 5명이 인식 되는 경우
        #print("name_xy_coord: ", name_xy_coord)
        # x 좌표를 기준으로 오름차순 정렬
        name_xy_coord = dict(
            sorted(name_xy_coord.items(), key=lambda x: x[1][0]))

        names_list = list(name_xy_coord.items())

        left_names_list = names_list[0:2]
        left_names_list = sorted(left_names_list, key=lambda x: x[1][1])
        right_names_list = names_list[3:5]
        right_names_list = sorted(right_names_list, key=lambda x: x[1][1])

        name_xy_coord = {}
        name_xy_coord[names_list[2][0]] = 2

        pos = 1
        for left_info in left_names_list:
            name_xy_coord[left_info[0]] = pos
            pos = 4

        pos = 3
        for right_info in right_names_list:
            name_xy_coord[right_info[0]] = pos
            pos = 5

    elif len(name_xy_coord) == 4:  # 4명만 인식된 경우
        name_xy_coord = dict(
            sorted(name_xy_coord.items(), key=lambda x: x[1][0]))

        names_list = list(name_xy_coord.items())

        left_names_list = names_list[0:2]
        left_names_list = sorted(left_names_list, key=lambda x: x[1][1])
        right_names_list = names_list[2:4]
        right_names_list = sorted(right_names_list, key=lambda x: x[1][1])

        name_xy_coord = {}
        pos = 1
        for left_info in left_names_list:
            name_xy_coord[left_info[0]] = pos
            pos = 4

        pos = 3
        for right_info in right_names_list:
            name_xy_coord[right_info[0]] = pos
            pos = 5

    elif len(name_xy_coord) == 3:  # 3명만 인식된 경우
        name_xy_coord = dict(
            sorted(name_xy_coord.items(), key=lambda x: x[1][0]))

        names_list = list(name_xy_coord.items())

        name_xy_coord = {}

        lx, ly = names_list[0][0], names_list[0][1]
        mx, my = names_list[1][0], names_list[1][1]
        rx, ry = names_list[2][0], names_list[2][1]

        if ((lx - mx) ** 2 + (ly - my) ** 2) ** (1/2) > ((rx - mx) ** 2 + (ry - my) ** 2) ** (1/2):
            name_xy_coord[names_list[1]] = 5
        else:
            name_xy_coord[names_list[1]] = 4

        name_xy_coord[names_list[0]] = 1
        name_xy_coord[names_list[2]] = 3

    elif len(name_xy_coord) == 2:  # 2명만 인식된 경우
        name_xy_coord = dict(
            sorted(name_xy_coord.items(), key=lambda x: x[1][0]))

        names_list = list(name_xy_coord.items())

        name_xy_coord = {}
        name_xy_coord[names_list[0]] = 1
        name_xy_coord[names_list[1]] = 3

    elif len(name_xy_coord) == 1:  # 1명만 인식된 경우
        name_xy_coord = dict(
            sorted(name_xy_coord.items(), key=lambda x: x[1][0]))

        names_list = list(name_xy_coord.items())

        name_xy_coord = {}
        name_xy_coord[names_list[0]] = 2

    return name_xy_coord


def guess_real_pos(lp, rp, r, base_line_length):
    p1 = (lp - 1220 + 5760) % 5760 * 2 * r * math.pi / 5760
    p2 = (rp - 1220 + 5760) % 5760 * 2 * r * math.pi / 5760

    try:
        a1 = -r * math.cos(p1 / r) - base_line_length / 2
        b1 = r * math.sin(p1 / r)
        a2 = -r * math.cos(p2 / r) + base_line_length / 2
        b2 = r * math.sin(p2 / r)

        inter_x, inter_y = get_intersect_point(
            a1, b1, -base_line_length/2, 0, a2, b2, base_line_length/2, 0)
        return int(inter_x), int(inter_y)

    except:
        return None, None


@app.route('/video_info_streaming')
def video_feed():
    # 왼쪽 카메라 객체 생성

    # 주 카메라. 보통 왼쪽  # 얼굴 사라지는것, 손 검출
    main_cam = Camera360("demo_video/Face_hand_left.mp4")
    # 주 카메라. 보통 왼쪽 카메라 # 스테레오
    #main_cam = Camera360("demo_video/Stereo_left_2.mp4")
    # 다른쪽(오른쪽) 카메라 객체 생성
    # 주 카메라. 보통 오른쪽  # 얼굴 사라지는것, 손 검출
    sub_cam = Camera360("demo_video/Face_hand_right.mp4")
    # 주 카메라. 보통 오른쪽 카메라 # 스테레오
    #sub_cam = Camera360("demo_video/Stereo_right_2.mp4")

    def generate_info():
        main_frame_idx, sub_frame_idx = 0, 0
        while True:
            main_frame_idx += 1
            sub_frame_idx += 1
            # time.sleep(1)
            # [재실 여부, 손 업/다운, 얼굴의 x좌표(픽셀단위), 왼쪽부터 몇 번째 위치하는 얼굴인지]
            main_cam_info = main_cam.get_frame_main()
            img_pos_info = {}
            for name in main_cam_info.keys():  # 현태, 진호 ... 5개
                # MTCNN 이후 왼쪽부터 몇번째에 위치하는 얼굴인지
                img_pos_info[name] = main_cam_info[name][3]

            # print('정렬 전 img_pos_info: ', img_pos_info)
            img_pos_info = dict(
                sorted(img_pos_info.items(), key=lambda x: x[1]))
            # print('정렬 후 img_pos_info: ', img_pos_info)

            print((main_frame_idx) % 5, sub_frame_idx, main_cam_info)
            if main_frame_idx % 5 == 0:  # 10 프레임마다 재실 여부 업데이트해서 보냄
                if sub_frame_idx % 1033 == 0:  # 스테레오 매칭 진행 시작
                    sub_cam_info = sub_cam.get_frame_sub(
                        img_pos_info)  # 6080px 중 왼쪽부터 몇 번째 픽셀에 있는 지
                    print("30프레임째 스테레오 매칭 진행 sub_cam_info", sub_cam_info)
                    name_xy_coord = {}  # 사람 신원 별 위치 좌표 (x, y) 기록 좌표
                    for name in sub_cam_info.keys():
                        if sub_cam_info[name] == 9999:
                            continue
                        x, y = guess_real_pos(
                            main_cam_info[name][2], sub_cam_info[name], 5, 84)
                        print(name, x, y)
                        if x is None or y is None:
                            continue

                        name_xy_coord[name] = (x, y)

                    name_xy_coord = get_position_in_classroom(name_xy_coord)
                    print(name_xy_coord)

                    for name in name_xy_coord.keys():
                        main_cam_info[name][2] = name_xy_coord[name]

                    # main_cam_info.append('str-mat-o') # 스테레오 매칭을 진행했다는 플래그, idx 4 --> 자바스크립트에 적용
                    main_cam_info['is_stereo_matched'] = 'o'
                    main_cam_info["joohyeong"][2] = 2
                    print('60000프레임째 스테레오 매칭 진행 main_cam_info: ', main_cam_info)

                    json_data = json.dumps(main_cam_info)  # 딕셔너리 -> json
                    yield f"data:{json_data}\n\n"  # json 스트리밍
                    time.sleep(10)
                else:
                    # main_cam_info.append('str-mat-x') # 스테레오 매칭을 진행하지 않았다는 플래그
                    main_cam_info['is_stereo_matched'] = 'x'
                    main_cam_info['joohyeong'][0] = 'o'
                    main_cam_info['joohyeong'][1] = 'd'
                    main_cam_info['jaehyeon'][1] = 'd'
                    main_cam_info['hyeontae'][1] = 'u'

                    sub_cam.skip_frame()  # sub cam은 스테레오 매칭 시에만 이용되고 그 이전에는 skip
                    json_data = json.dumps(main_cam_info)  # 딕셔너리 -> json
                    yield f"data:{json_data}\n\n"  # json 스트리밍
                    time.sleep(1)

                # main_cam.update_cam("https://youtu.be/qDiXBc8y1TA") #왼쪽 카메라 업데이트
                # sub_cam.update_cam("https://youtu.be/JoSs13xAv2g") #오른쪽 카메라 업데이트

            else:
                sub_cam.skip_frame()
    return Response(generate_info(), mimetype="text/event-stream")


if __name__ == '__main__':
    app.run(debug=True)
