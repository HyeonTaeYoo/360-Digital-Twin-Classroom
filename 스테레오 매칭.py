from mtcnn import MTCNN
from keras.models import Model, load_model
from PIL import Image
import cv2
import math
import numpy as np
import model_prediction
import matplotlib.pyplot as plt
import operator

detector = MTCNN()
targetX, targetY = 100, 100
# 학생 명단 // model_prediction.py에 정의된 것 이용
categories = model_prediction.categories
left_pos = {categories[0]: 0, categories[1]: 0,
            categories[2]: 0, categories[3]: 0, categories[4]: 0}
right_pos = {categories[0]: 0, categories[1]: 0,
             categories[2]: 0, categories[3]: 0, categories[4]: 0}


def square_face(img, box_x, box_y, box_width, box_height):
    try:
        face_center_x = box_x + int(1/2 * box_width)
        face_center_y = box_y + int(1/2 * box_height)
        a = int(1/2 * box_width) if box_width > box_height else int(1/2 * box_height)
        # 최대한 얼굴 형태 유지하며 정방형으로 잘라냄
        face = img[face_center_y - a: face_center_y +
                   a, face_center_x - a: face_center_x + a]
        # 얼굴 (100, 100)로 만듦
        face = cv2.resize(face, dsize=(targetX, targetY))

        return face

    except:
        return None


def face_pos_left(img):  # 한 img에서 얼굴과 그 얼굴의 신원을 파악하는 코드
    face_infos = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #print(len(face_infos)) # 발견된 얼굴의 수

    print(type(face_infos))
    print(face_infos)
    print("===========================================")
    face_infos = sorted(face_infos, key=lambda x: x['box'][0])
    print(face_infos)

    for face_info in face_infos:
        box = face_info['box']
        box_x, box_y = box[0], box[1]
        box_width, box_heigth = box[2], box[3]

        # 얼굴 중심을 기준으로 정방형으로 얼굴 추출
        face = square_face(img, box_x, box_y, box_width, box_heigth)

        if face is None:
            continue  # 얼굴이 아닌 경우에는 그냥 지나침

        # 모델을 이용한 신원 파악 및 정확도
        # MTCNN이나 Keras에서는 이미지를 PILLOW로 처리하는데, 현재 코드는
        # OpenCV 즉, ndarray 이용하므로 채널과 자료형 변환
        name, acc = model_prediction.predict_by_model(Image.fromarray(
            cv2.cvtColor(face, cv2.COLOR_BGR2RGB)), categories)
        left_pos[name] = box_x + box_width // 2

        # # 얼굴에 바운더리 박스 그리기
        img = cv2.rectangle(img, (box_x, box_y), (box_x +
                            box_width, box_y + box_heigth), (255, 0, 0), 3)
        # # 박스 위에 이름과 정확도 출력하기
        # img = cv2.putText(
        #                  img, name + ' '+str(acc), (box_x, box_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # 이미지에서 현재 파악하고 있는 얼굴의 신원과 정확도 반환
        print(name, acc, box_x + box_width // 2)
        dummy_img = cv2.resize(img, None, fx=0.3, fy=0.3)
        cv2.imshow("img", dummy_img)
        cv2.waitKey()
    return img


def face_pos_right(img):  # 한 img에서 얼굴과 그 얼굴의 신원을 파악하는 코드
    face_infos = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #print(len(face_infos)) # 발견된 얼굴의 수

    for face_info in face_infos:
        box = face_info['box']
        box_x, box_y = box[0], box[1]
        box_width, box_heigth = box[2], box[3]

        # 얼굴 중심을 기준으로 정방형으로 얼굴 추출
        face = square_face(img, box_x, box_y, box_width, box_heigth)

        if face is None:
            continue  # 얼굴이 아닌 경우에는 그냥 지나침

        # 모델을 이용한 신원 파악 및 정확도
        # MTCNN이나 Keras에서는 이미지를 PILLOW로 처리하는데, 현재 코드는
        # OpenCV 즉, ndarray 이용하므로 채널과 자료형 변환
        name, acc = model_prediction.predict_by_model(Image.fromarray(
            cv2.cvtColor(face, cv2.COLOR_BGR2RGB)), categories)
        right_pos[name] = box_x + box_width // 2
        #print(box_x + box_width // 2)

        # # 얼굴에 바운더리 박스 그리기
        img = cv2.rectangle(img, (box_x, box_y), (box_x +
                            box_width, box_y + box_heigth), (255, 0, 0), 3)
        # # 박스 위에 이름과 정확도 출력하기
        # img = cv2.putText(
        #                  img, name + ' '+str(acc), (box_x, box_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # 이미지에서 현재 파악하고 있는 얼굴의 신원과 정확도 반환
        print(name, acc, box_x + box_width // 2)
        dummy_img = cv2.resize(img, None, fx=0.3, fy=0.3)
        cv2.imshow("img", dummy_img)
        cv2.waitKey()
    return img


def get_intersect_point(a1, b1, c1, d1, a2, b2, c2, d2):
    x = (b2 - b1 + a1 * ((b1 - d1) / (a1 - c1)) - a2 *
         ((b2-d2) / (a2-c2))) / ((b1-d1)/(a1-c1) - (b2-d2) / (a2 - c2))
    y = ((b1 - d1) / (a1 - c1)) * (x - a1) + b1

    return x, y


def guess_real_pos(lp, rp, r, base_line_length):
    p1 = (lp - 1288 + 6080) % 6080 * 2 * r * math.pi / 6080
    p2 = (rp - 1288 + 6080) % 6080 * 2 * r * math.pi / 6080

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


def rotate_pt(x, y):
    return math.cos(76 * math.pi / 180) * x - math.sin(76 * math.pi / 180) * y, math.sin(76 * math.pi / 180) * x + math.cos(76 * math.pi / 180) * y


def align_pts(dict, left, right):
    print(dict)
    x2, y2 = left[0], left[1]
    x1, y1 = right[0], right[1]

    a = ((x2 - x1)**2 + (y2 - y1)**2) ** (1/2)

    for i in list(dict.keys()):
        x, y = dict[i][0], dict[i][1]
        print(x)
        x_prime = abs(x1 - x2) / a * (x - x1) - (y2 - y1) / a * (y - y1)
        y_prime = (y2 - y1) / a * (x - x1) + abs(x1 - x2) / a * (y - y1)
        dict[i][0], dict[i][1] = x_prime, y_prime

    return dict


if __name__ == '__main__':
    print("왼쪽 이미지에서 얼굴 인식 결과")
    face_pos_left(cv2.imread('./stereo_imgs/0602_pano_left1.jpg'))
    print("오른쪽 이미지에서 얼굴 인식 결과")
    face_pos_right(cv2.imread("./stereo_imgs/0602_pano_right1.jpg"))

    base_line = 84
    print("base_line 길이: ", base_line)

    jinho_left_pos = left_pos['jinho']
    jinho_right_pos = right_pos['jinho']

    yoosung_left_pos = left_pos['yoosung']
    yoosung_right_pos = right_pos['yoosung']

    hyeontae_left_pos = left_pos['hyeontae']
    hyeontae_right_pos = right_pos['hyeontae']

    joohyeong_left_pos = left_pos['joohyeong']
    joohyeong_right_pos = right_pos['joohyeong']

    jaehyeon_left_pos = left_pos['jaehyeon']
    jaehyeon_right_pos = right_pos['jaehyeon']

    print()
    print("jinho:", jinho_left_pos, jinho_right_pos)
    print("joohyeong:", joohyeong_left_pos, joohyeong_right_pos)
    print("hyeontae:", hyeontae_left_pos, hyeontae_right_pos)
    print("yoosung:", yoosung_left_pos, yoosung_right_pos)
    print("jaehyeon:", jaehyeon_left_pos, jaehyeon_right_pos)
    print()

    x_coords = []
    y_coords = []

    pts = {}
    if hyeontae_left_pos != 0 and hyeontae_right_pos != 0:
        hyeontae_x, hyeontae_y = guess_real_pos(
            hyeontae_left_pos, hyeontae_right_pos, 5, base_line)
        if hyeontae_x != None and hyeontae_y != None:
            x_coords.append(hyeontae_x)
            y_coords.append(hyeontae_y)
            print('hyeontae pos:', hyeontae_x, hyeontae_y)
            pts['hyeontae'] = [hyeontae_x, hyeontae_y]
        else:
            pass

    if jaehyeon_left_pos != 0 and jaehyeon_right_pos != 0:
        jaehyeon_x, jaehyeon_y = guess_real_pos(
            jaehyeon_left_pos, jaehyeon_right_pos, 5, base_line)
        if jaehyeon_x != None and jaehyeon_y != None:
            x_coords.append(jaehyeon_x)
            y_coords.append(jaehyeon_y)
            print('jaehyeon pos:', jaehyeon_x, jaehyeon_y)
            pts['jaehyeon'] = [jaehyeon_x, jaehyeon_y]
        else:
            pass

    if jinho_left_pos != 0 and jinho_right_pos != 0:
        jinho_x, jinho_y = guess_real_pos(
            jinho_left_pos, jinho_right_pos, 5, base_line)
        if jinho_x != None and jinho_y != None:
            x_coords.append(jinho_x)
            y_coords.append(jinho_y)
            print('jinho pos:', jinho_x, jinho_y)
            pts['jinho'] = [jinho_x, jinho_y]
        else:
            pass

    if joohyeong_left_pos != 0 and joohyeong_right_pos != 0:
        joohyeong_x, joohyeong_y = guess_real_pos(
            joohyeong_left_pos, joohyeong_right_pos, 5, base_line)
        if joohyeong_x != None and joohyeong_y != None:
            x_coords.append(joohyeong_x)
            y_coords.append(joohyeong_y)
            print('joohyeong pos:', joohyeong_x, joohyeong_y)
            pts['joohyeong'] = [joohyeong_x, joohyeong_y]
        else:
            pass

    if yoosung_left_pos != 0 and yoosung_right_pos != 0:
        yoosung_x, yoosung_y = guess_real_pos(
            yoosung_left_pos, yoosung_right_pos, 5, base_line)
        if yoosung_x != None and yoosung_y != None:
            x_coords.append(yoosung_x)
            y_coords.append(yoosung_y)
            print('yoosung pos:', yoosung_x, yoosung_y)
            pts['yoosung'] = [yoosung_x, yoosung_y]
        else:
            pass

    plt.plot([-base_line / 2, base_line / 2], [0, 0], 'bo')  # 카메라 좌표 추가
    plt.plot(x_coords, y_coords, 'ro')  # 스테레오 매칭 점들 추가
    plt.ylim(-450, 450)
    plt.xlim(-450, 450)

    # print(pts)
    # print(pts['hyeontae'][0])

    # new_pts = sorted(pts.items(), key=lambda x : x[1][0])
    # new_pts_y = sorted(pts.items(), key=lambda x : x[1][1])

    # most_left_pt = new_pts[0][1]
    # most_right_pt = new_pts_y[0][1]

    # print(most_left_pt)
    # print(most_right_pt)

    # aligned_pts = align_pts(pts, most_left_pt, most_right_pt)

    # new_x_coords = []
    # new_y_coords = []

    # for i in list(aligned_pts.keys()):
    #     new_x_coords.append(aligned_pts[i][0])
    #     new_y_coords.append(aligned_pts[i][1])

    # print(aligned_pts)

    # plt.plot([-base_line / 2, base_line / 2], [0, 0], 'bo') # 카메라 좌표 추가
    # plt.plot(new_x_coords, new_y_coords, 'ro') # 스테레오 매칭 점들 추가

    plt.show()
