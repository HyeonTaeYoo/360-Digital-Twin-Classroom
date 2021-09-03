# from mtcnn import MTCNN
# from keras.models import Model, load_model
# from PIL import Image
# import cv2
# import math
# import numpy as np
# import model_prediction
# from collections import deque

# '''여기서는 임의의 이미지를 사용하지만
# 실제로 사용될 이미지는 실시간이든 비디오든, 어떤 프레임임'''

# detector = MTCNN()
# targetX, targetY = 100, 100
# # 학생 명단 // model_prediction.py에 정의된 것 이용
# categories = model_prediction.categories

# # 입 모양 및 발화 상태 파악을 위한 모델과 빈 딕셔너리
# IMG_SIZE = (34, 26)  # 입 이미지의 가로, 세로 사이즈
# class_participants = {}
# check_list = {categories[0]: deque('x'*10, maxlen=10), categories[1]: deque('x'*10, maxlen=10), categories[2]: deque(
#     'x'*10, maxlen=10), categories[3]: deque('x'*10, maxlen=10), categories[4]: deque('x'*10, maxlen=10)}


# def make_mask_image(img):
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 여기 RGB 인지 BGR 인지 체크

#     # img_h,img_s,img_v = cv.split(img_hsv)

#     low = (0, 30, 0)
#     high = (15, 255, 255)

#     img_mask = cv2.inRange(img_hsv, low, high)
#     return img_mask


# def findMaxArea(contours, box_x, box_y, width, height):

#     max_contour = None
#     max_area = -1

#     for contour in contours:

#         x, y, w, h = cv2.boundingRect(contour)
#         area = 0
#         if box_x-(width*2) <= x and (box_y - height) <= y and (box_x)-30 > x and (box_y + 2*height) >= y:
#             area = cv2.contourArea(contour)

#         if (w*h)*0.4 > area:
#             continue

#         if w > h:
#             continue

#         if area > max_area:
#             max_area = area
#             max_contour = contour

#     if max_area < 1000:
#         max_area = -1

#     return max_area, max_contour


# def square_face(img, box_x, box_y, box_width, box_height):
#     try:
#         face_center_x = box_x + int(1/2 * box_width)
#         face_center_y = box_y + int(1/2 * box_height)
#         a = int(1/2 * box_width) if box_width > box_height else int(1/2 * box_height)
#         # 최대한 얼굴 형태 유지하며 정방형으로 잘라냄
#         face = img[face_center_y - a: face_center_y +
#                    a, face_center_x - a: face_center_x + a]
#         # 얼굴 (100, 100)로 만듦
#         face = cv2.resize(face, dsize=(targetX, targetY))

#         return face

#     except:
#         return None


# def face_catcher(img):  # 한 img에서 얼굴과 그 얼굴의 신원을 파악하는 코드
#     name_list = {categories[0]: ['x', 'd', -100, 999], categories[1]: ['x', 'd', -100, 999], categories[2]: [
#         'x', 'd', -100, 999], categories[3]: ['x', 'd', -100, 999], categories[4]: ['x', 'd', -100, 999]}

#     pos_checked = {categories[0]: [False, 0], categories[1]: [False, 0], categories[2]: [
#         False, 0], categories[3]: [False, 0], categories[4]: [False, 0]}
#     acc_checked = {categories[0]: 0, categories[1]: 0,
#                    categories[2]: 0, categories[3]: 0, categories[4]: 0}
#     face_infos = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#     ###손드는 것 검출 코드###
#     # STEP1
#     img_result = img.copy()  # 손들기 검출을 위한 이미지 복사

#     # STEP 2
#     img_binary = make_mask_image(img)

#     # STEP 3
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, 1)

#     # STEP 4
#     contours, hierarchy = cv2.findContours(
#         img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # face info 정렬 코드

#     # print(type(face_infos))
#     # print(face_infos)
#     print("===========================================")
#     face_infos = sorted(face_infos, key=lambda x: x['box'][0])

#     for i, name in enumerate(face_infos):
#         box = name['box']
#         box_x, box_y = box[0], box[1]
#         box_width, box_height = box[2], box[3]

#         # 얼굴 중심을 기준으로 정방형으로 얼굴 추출
#         face = square_face(img, box_x, box_y, box_width, box_height)

#         if face is None:
#             continue  # 얼굴이 아닌 경우에는 그냥 지나침

#         # 모델을 이용한 신원 파악 및 정확도
#         # MTCNN이나 Keras에서는 이미지를 PILLOW로 처리하는데, 현재 코드는
#         # OpenCV 즉, ndarray 이용하므로 채널과 자료형 변환
#         names, acc = model_prediction.predict_by_model(Image.fromarray(
#             cv2.cvtColor(face, cv2.COLOR_BGR2RGB)), categories)

#         ################## 손드는 것 감지 하는 코드 #########################
#         # 각 사람마다 손들 공간 표시
#         img = cv2.rectangle(img, (box_x - 2*box_width, box_y-box_height),
#                             (box_x-30, box_y + 2*box_height), (0, 255, 0), 2)

#         # STEP 5
#         max_area, max_contour = findMaxArea(
#             contours, box_x, box_y, box_width, box_height)

#         if max_area >= 1000:
#             # cv2.putText(img,"Hand up",(box_x- box_width,box_y),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA )
#             name_list[names][1] = 'u'
#         else:
#             name_list[names][1] = 'd'

#         # =================================================================================================================
#         # 사람 위치 표시

#         if pos_checked[names][0] is False:  # 아직 위치가 체크되지 않은 사람이라면
#             name_list[names][2] = box_x + box_width // 2
#             pos_checked[names][0] = True
#             pos_checked[names][1] = acc

#         else:  # 이미 위치가 체크된 적 있는 사람이라면
#             if pos_checked[names][1] < acc:
#                 name_list[names][2] = box_x + box_width // 2
#                 pos_checked[names][1] = acc
#         # =================================================================================================================

#         # print(name, max_area)
#         # print("")
#         # cv2.drawContours(img, [max_contour], 0, (0, 0, 255), 3)
#         # cv2.imshow("Result", img_result)

#         # # 얼굴에 바운더리 박스 그리기
#         # img = cv2.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 0, 0), 1)
#         # # 박스 위에 이름과 정확도 출력하기
#         # img = cv2.putText(
#         #                 img, name + ' '+str(acc), (box_x, box_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

#         # name_list[name]='o' # 발화자가 아니라, {'jinho':'x', 'jahyeon' :'o'}
#         # 정확도 60이상시에 이사람이 맞다라고 체크
#         print("aaaa")
#         print(names)
#         print(acc)
#         if name_list[names][3] == 999:
#             name_list[names][3] = i

#         if acc_checked[names] < acc:
#             name_list[names][3] = i
#             acc_checked[names] = acc

#         if acc > 60:
#             name_list[names][0] = 'o'

#         # print(name, acc)

#     # 실제 출석여부를 반환할 list를 수정
#     # 현재 프레임에서 몇명이 인식됐는지 리스트가 수정됨.
#     # 덱에 현재 인식여부를 추가함
#     # 그 추가된 후에 개수를 파악해서 조건을 만족하면 출석과 비출석을 정해서 최종 리턴

#     for names in name_list.keys():
#         check_list[names].append(name_list[names][0])
#         if check_list[names].count('o') >= 6:
#             name_list[names][0] = 'o'
#         else:
#             name_list[names][0] = 'x'

#     # for i, name in enumerate(face_infos):
#     #     print(str(i) + " 번째 위치 " + str(name))
#     #     name_list[names][3] = i
#     print(acc_checked)
#     return name_list

# # i=0
# # if __name__ == '__main__':
# #     left_img = cv2.imread("360_streo_left.jpg")
# #     right_img = cv2.imread("360_streo_right.jpg")
# #     while True:
# #         print(face_catcher(right_img))
# #         i +=1
# #         if i==6:
# #             break
from mtcnn import MTCNN
from keras.models import Model, load_model
from PIL import Image
import cv2
import math
import numpy as np
import model_prediction
from collections import deque

'''여기서는 임의의 이미지를 사용하지만
실제로 사용될 이미지는 실시간이든 비디오든, 어떤 프레임임'''

detector = MTCNN()
targetX, targetY = 100, 100
# 학생 명단 // model_prediction.py에 정의된 것 이용
categories = model_prediction.categories

# 입 모양 및 발화 상태 파악을 위한 모델과 빈 딕셔너리
IMG_SIZE = (34, 26)  # 입 이미지의 가로, 세로 사이즈
#mouth_model = load_model('mouth_models/2021_04_02_01_57_41.h5')
class_participants = {}
check_list = {categories[0]: deque('x'*10, maxlen=10), categories[1]: deque('x'*10, maxlen=10), categories[2]: deque(
    'x'*10, maxlen=10), categories[3]: deque('x'*10, maxlen=10), categories[4]: deque('x'*10, maxlen=10)}


def make_mask_image(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 여기 RGB 인지 BGR 인지 체크

    #img_h,img_s,img_v = cv.split(img_hsv)

    low = (0, 30, 0)
    high = (15, 255, 255)

    img_mask = cv2.inRange(img_hsv, low, high)
    return img_mask


def findMaxArea(contours, box_x, box_y, width, height):

    max_contour = None
    max_area = -1

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        area = 0
        if box_x-(width*2) <= x and (box_y - height) <= y and (box_x)-30 > x and (box_y + 2*height) >= y:
            area = cv2.contourArea(contour)

        if (w*h)*0.4 > area:
            continue

        if w > h:
            continue

        if area > max_area:
            max_area = area
            max_contour = contour

    if max_area < 1000:
        max_area = -1

    return max_area, max_contour


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


def face_catcher_main(img):  # 한 img에서 얼굴과 그 얼굴의 신원을 파악하는 코드
    name_list = {categories[0]: ['x', 'd', -100, 999], categories[1]: ['x', 'd', -100, 999], categories[2]: [
        'x', 'd', -100, 999], categories[3]: ['x', 'd', -100, 999], categories[4]: ['x', 'd', -100, 999]}

    pos_checked = {categories[0]: [False, 0], categories[1]: [False, 0], categories[2]: [
        False, 0], categories[3]: [False, 0], categories[4]: [False, 0]}
    acc_checked = {categories[0]: 0, categories[1]: 0,
                   categories[2]: 0, categories[3]: 0, categories[4]: 0}
    face_infos = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ###손드는 것 검출 코드###
    # STEP1
    img_result = img.copy()  # 손들기 검출을 위한 이미지 복사

    # STEP 2
    img_binary = make_mask_image(img)

    # STEP 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, 1)

    # STEP 4
    contours, hierarchy = cv2.findContours(
        img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # face info 정렬 코드

    # print(type(face_infos))
    # print(face_infos)

    face_infos = sorted(face_infos, key=lambda x: x['box'][0])

    for i, name in enumerate(face_infos):
        box = name['box']
        box_x, box_y = box[0], box[1]
        box_width, box_height = box[2], box[3]

        # 얼굴 중심을 기준으로 정방형으로 얼굴 추출
        face = square_face(img, box_x, box_y, box_width, box_height)

        if face is None:
            continue  # 얼굴이 아닌 경우에는 그냥 지나침

        # 모델을 이용한 신원 파악 및 정확도
        # MTCNN이나 Keras에서는 이미지를 PILLOW로 처리하는데, 현재 코드는
        # OpenCV 즉, ndarray 이용하므로 채널과 자료형 변환
        names, acc = model_prediction.predict_by_model(Image.fromarray(
            cv2.cvtColor(face, cv2.COLOR_BGR2RGB)), categories)
        print(names, acc)

        ################## 손드는 것 감지 하는 코드 #########################
        # 각 사람마다 손들 공간 표시
        img = cv2.rectangle(img, (box_x - 2*box_width, box_y-box_height),
                            (box_x-30, box_y + 2*box_height), (0, 255, 0), 2)

        # STEP 5
        max_area, max_contour = findMaxArea(
            contours, box_x, box_y, box_width, box_height)

        if max_area >= 100:
            cv2.putText(img, "Hand up", (box_x - box_width, box_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

            name_list[names][1] = 'u'
        else:
            name_list[names][1] = 'd'

        # =================================================================================================================
        # 사람 위치 표시

        if pos_checked[names][0] is False:  # 아직 위치가 체크되지 않은 사람이라면
            name_list[names][2] = box_x + box_width // 2
            pos_checked[names][0] = True
            pos_checked[names][1] = acc

        else:  # 이미 위치가 체크된 적 있는 사람이라면
            if pos_checked[names][1] < acc:
                name_list[names][2] = box_x + box_width // 2
                pos_checked[names][1] = acc
        # =================================================================================================================

        #print(name, max_area)
        # print("")
        #cv2.drawContours(img, [max_contour], 0, (0, 0, 255), 3)
        #cv2.imshow("Result", img_result)

        # # 얼굴에 바운더리 박스 그리기
        # img = cv2.rectangle(img, (box_x, box_y), (box_x +
        #                                           box_width, box_y + box_height), (255, 0, 0), 1)
        # print('-----------------------------', name, acc)
        # cv2.imshow("dddd", img)
        # cv2.waitKey(0)
        # # 박스 위에 이름과 정확도 출력하기
        # img = cv2.putText(
        #                 img, name + ' '+str(acc), (box_x, box_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # name_list[name]='o' # 발화자가 아니라, {'jinho':'x', 'jahyeon' :'o'}
        # 정확도 60이상시에 이사람이 맞다라고 체크

        if name_list[names][3] == 999:
            name_list[names][3] = i

        if acc_checked[names] < acc:
            name_list[names][3] = i
            acc_checked[names] = acc

        if acc > 60:
            name_list[names][0] = 'o'

        # print(name, acc)

    # 실제 출석여부를 반환할 list를 수정
    # 현재 프레임에서 몇명이 인식됐는지 리스트가 수정됨.
    # 덱에 현재 인식여부를 추가함
    # 그 추가된 후에 개수를 파악해서 조건을 만족하면 출석과 비출석을 정해서 최종 리턴

    for names in name_list.keys():
        check_list[names].append(name_list[names][0])
        if check_list[names].count('o') >= 3:
            name_list[names][0] = 'o'
        else:
            name_list[names][0] = 'x'

    # for i, name in enumerate(face_infos):
    #     print(str(i) + " 번째 위치 " + str(name))
    #     name_list[names][3] = i

    # img = cv2.resize(img, dsize=(0,0), fx = 0.3, fy = 0.3)
    # cv2.imshow("img", img)
    # cv2.waitKey()
    return name_list


def face_catcher_sub(img, img_pos_info):  # 한 img에서 얼굴과 그 얼굴의 신원을 파악하는 코드
    # img_pos_info는 MTCNN 결과 기준으로 0, 1, 2,3 ... 이렇게 정렬된 상태
    img_pos_info_name_list = list(img_pos_info.keys())
    pos_info = {}
    face_infos = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 얘도 MTCNN 기준으로 정렬된 것
    face_infos = sorted(face_infos, key=lambda x: x['box'][0])
    for i, face_info in enumerate(face_infos):  # 웬만하면 5번 돌아갈것
        if img_pos_info[img_pos_info_name_list[i]] == 999:
            pos_info[img_pos_info_name_list[i]] = 9999
        else:
            box = face_info['box']
            box_x, box_y = box[0], box[1]
            box_width, box_height = box[2], box[3]
            pos_info[img_pos_info_name_list[i]] = box_x + box_width // 2

    return pos_info


def face_catcher(img):  # 한 img에서 얼굴과 그 얼굴의 신원을 파악하는 코드
    name_list = {categories[0]: ['x', 'd', -100], categories[1]: ['x', 'd', -100], categories[2]
        : ['x', 'd', -100], categories[3]: ['x', 'd', -100], categories[4]: ['x', 'd', -100]}
    pos_checked = {categories[0]: [False, 0], categories[1]: [False, 0], categories[2]: [
        False, 0], categories[3]: [False, 0], categories[4]: [False, 0]}
    face_infos = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # print(len(face_infos)) # 발견된 얼굴의 수

    ###손드는 것 검출 코드###
    # STEP1
    img_result = img.copy()  # 손들기 검출을 위한 이미지 복사

    # STEP 2
    img_binary = make_mask_image(img)

    # STEP 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, 1)

    # STEP 4
    contours, hierarchy = cv2.findContours(
        img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # face info 정렬 코드
    # print(type(face_infos))
    # print(face_infos)
    # print("===========================================")
    # face_infos = sorted(face_infos, key=lambda x : x['box'][0])
    # print(face_infos)

    for face_info in face_infos:
        box = face_info['box']
        box_x, box_y = box[0], box[1]
        box_width, box_height = box[2], box[3]

        # 얼굴 중심을 기준으로 정방형으로 얼굴 추출
        face = square_face(img, box_x, box_y, box_width, box_height)

        if face is None:
            continue  # 얼굴이 아닌 경우에는 그냥 지나침

        # 모델을 이용한 신원 파악 및 정확도
        # MTCNN이나 Keras에서는 이미지를 PILLOW로 처리하는데, 현재 코드는
        # OpenCV 즉, ndarray 이용하므로 채널과 자료형 변환
        name, acc = model_prediction.predict_by_model(Image.fromarray(
            cv2.cvtColor(face, cv2.COLOR_BGR2RGB)), categories)

        ################## 손드는 것 감지 하는 코드 #########################
        # 각 사람마다 손들 공간 표시
        img = cv2.rectangle(img, (box_x - 2*box_width, box_y-box_height),
                            (box_x-30, box_y + 2*box_height), (0, 255, 0), 2)

        # STEP 5
        max_area, max_contour = findMaxArea(
            contours, box_x, box_y, box_width, box_height)

        if max_area >= 1000:
            # cv2.putText(img,"Hand up",(box_x- box_width,box_y),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA )
            name_list[name][1] = 'u'
        else:
            name_list[name][1] = 'd'

        # =================================================================================================================
        # 사람 위치 표시
        if pos_checked[name][0] is False:  # 아직 위치가 체크되지 않은 사람이라면
            name_list[name][2] = box_x + box_width // 2
            pos_checked[name][0] = True
            pos_checked[name][1] = acc

        else:  # 이미 위치가 체크된 적 있는 사람이라면
            if pos_checked[name][1] < acc:
                name_list[name][2] = box_x + box_width // 2
                pos_checked[name][1] = acc
        # =================================================================================================================

        # print(name, max_area)
        # print("")
        # cv2.drawContours(img, [max_contour], 0, (0, 0, 255), 3)
        # cv2.imshow("Result", img_result)

        # # 얼굴에 바운더리 박스 그리기
        # img = cv2.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 0, 0), 1)
        # # 박스 위에 이름과 정확도 출력하기
        # img = cv2.putText(
        #                 img, name + ' '+str(acc), (box_x, box_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # name_list[name]='o' # 발화자가 아니라, {'jinho':'x', 'jahyeon' :'o'}
        # 정확도 60이상시에 이사람이 맞다라고 체크
        if acc > 60:
            name_list[name][0] = 'o'
        #print(name, acc)

    # 실제 출석여부를 반환할 list를 수정
    # 현재 프레임에서 몇명이 인식됐는지 리스트가 수정됨.
    # 덱에 현재 인식여부를 추가함
    # 그 추가된 후에 개수를 파악해서 조건을 만족하면 출석과 비출석을 정해서 최종 리턴
    for name in name_list.keys():
        check_list[name].append(name_list[name][0])
        if check_list[name].count('o') >= 6:
            name_list[name][0] = 'o'
        else:
            name_list[name][0] = 'x'

    return name_list


if __name__ == '__main__':
    cap = cv2.VideoCapture("Hand_left.MP4")
    # img = cv2.imread('./stereo_imgs/0604_pano_left_v1.jpg')
    # img2 = cv2.imread('./stereo_imgs/0604_pano_right_v1.jpg')
    ret, frame = cap.read()
    i = 0
    while True:
        ret, frame = cap.read()
        if ret is False:
            break

        print('====================')
        face_catcher_main(frame)
