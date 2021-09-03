from flask import Flask, render_template, Response
from camera_360 import Camera360
import time
import json
import frame_face_grab
import glob


app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/realSpatial')
def real():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/virtualSpatial')
def virtual():
    path = '.\\static\\ppt\\*'
    file_list = glob.glob(path)
    imglist = [file for file in file_list if file.endswith(".png")]
    imglist.sort()
    return render_template('webGL.html', imglist=imglist)


@app.route('/video_feed')
def video_feeding():
    return Response(gen(Camera360()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_info_streaming')
def video_feed():
    new_cam = Camera360()
    attendance_status = {}

    def generate_info():
        i = 0
        while True:
            i += 1
            # time.sleep(1)
            names = new_cam.get_frame()
            # print(i % 10, names)
            print("name" + str(names))
            # # if i%10==0:
            json_data = json.dumps(names)
            yield f"data:{json_data}\n\n"
            # print('go to sleep')
            # time.sleep(5)
            new_cam.update_cam()
    return Response(generate_info(), mimetype="text/event-stream")


if __name__ == '__main__':
    app.run(debug=True)
