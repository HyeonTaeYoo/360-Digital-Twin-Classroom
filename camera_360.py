from mtcnn import MTCNN
from PIL import Image
import cv2
import math
import model_prediction
import frame_face_grab
import pafy

ds_factor = 0.6
# img = cv2.imread('./stereo_imgs/0604_pano_left_v1.jpg')
# img2 = cv2.imread('./stereo_imgs/0604_pano_right_v1.jpg')


class Camera360(object):
    def __init__(self, url):
        # url = "https://youtu.be/Y76yRufNlmU"
        # video = pafy.new(url)
        # best = video.getbest(preftype="mp4")
        # self.video = cv2.VideoCapture(best.url)
        self.video = cv2.VideoCapture(url)

    def update_cam(self, url):
        self.video.release()
        #url = "https://youtu.be/Y76yRufNlmU"
        # video = pafy.new(url)
        # best = video.getbest(preftype="mp4")
        # self.video = cv2.VideoCapture(best.url)
        self.video = cv2.VideoCapture(0)

    def skip_frame(self):
        ret, frame = self.video.read()
        #cv2.imshow("ddfd", frame)
        # cv2.waitKey(1)

    def get_frame_main(self):
        ret, frame = self.video.read()  # 프레임 가져오고
        # print(frame.shape)
        cam_info = frame_face_grab.face_catcher_main(frame)  # [d,d,d,d]
        return cam_info

    def get_frame_sub(self, img_pos_info):
        ret, frame = self.video.read()  # 프레임 가져오고
        cam_info = frame_face_grab.face_catcher_sub(frame, img_pos_info)
        # 신원을 키로 갖는 딕셔너리 # value는 (10프레임 동안의 재실 여부, 손 업다운, 픽셀 위치, 영상 기준 몇번째 위치(1~5))
        return cam_info
