# 人脸数据集采集

import dlib         # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 Numpy
import cv2          # 图像处理的库 OpenCv

import os           # 读写文件
import shutil       # 读写文件

# Dlib 检测器
detector = dlib.get_frontal_face_detector()

# OpenCV 调用摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 480)

cnt_ss = 0                                      # 人脸截图的计数器
current_face_dir = ""                           # 存储人脸的文件夹
path_photos_from_camera = "data/data_faces/"    # 保存 faces images 的路径

# 新建保存人脸图像文件和数据CSV文件夹
def pre_work_mkdir():

    # 新建文件夹
    if os.path.isdir(path_photos_from_camera):
        pass
    else:
        os.mkdir(path_photos_from_camera)

pre_work_mkdir()

# 在之前 person_x 的序号按照 person_x+1 开始录入
if os.listdir("data/data_faces/"):
    # 获取已录入的最后一个人脸序号
    person_list = os.listdir("data/data_faces/")
    person_num_list = []
    for person in person_list:
        person_num_list.append(int(person.split('_')[-1]))
    person_cnt = max(person_num_list)
# 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入
else:
    person_cnt = 0

# 之后用来控制是否保存图像的 flag
save_flag = 1
# 之后用来检查是否先按 'n' 再按 's'
press_n_flag = 0

while cap.isOpened():
    flag, img_rd = cap.read()
    # 480  * 640

    kk = cv2.waitKey(1)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray, 0)       # 人脸数
    font = cv2.FONT_HERSHEY_COMPLEX     # 字体

    # 按下 'n' 新建存储人脸的文件夹
    if kk == ord('n'):
        person_cnt += 1
        current_face_dir = path_photos_from_camera + "person_" + str(person_cnt)
        os.makedirs(current_face_dir)
        print('\n')
        print("新建的人脸文件夹 / Create folders: ", current_face_dir)

        cnt_ss = 0              # 将人脸计数器清零
        press_n_flag = 1        # 已经按下 'n'

    # 检测到人脸
    if len(faces) != 0:
        # 矩形框
        for k, d in enumerate(faces):
            # 计算矩形大小(x,y), (宽度width, 高度height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # 计算矩形框大小
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())

            hh = int(height/2)
            ww = int(width/2)

            # 设置颜色 / the color of rectangle of faces detected
            color_rectangle = (255, 255, 255)

            # 判断人脸矩形框是否超出 480x640
            if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                cv2.putText(img_rd, "OUT OF RANGE", (20, 320), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                color_rectangle = (0, 0, 255)
                save_flag = 0
                if kk == ord('s'):
                    print("请调整位置 / Please adjust your position")
            else:
                color_rectangle = (255, 255, 255)
                save_flag = 1

            cv2.rectangle(img_rd,
                          tuple([d.left() - ww, d.top() - hh]),
                          tuple([d.right() + ww, d.bottom() + hh]),
                          color_rectangle, 2)

            # 根据人脸大小生成空的图像
            im_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

            if save_flag:
                # 按下 's' 保存摄像头中的人脸到本地
                if kk == ord('s'):
                    # 检查有没有先按'n'新建文件夹
                    if press_n_flag:
                        cnt_ss += 1
                        for ii in range(height*2):
                            for jj in range(width*2):
                                im_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                        cv2.imwrite(current_face_dir + "/img_face_" + str(cnt_ss) + ".jpg", im_blank)
                        print("写入本地 / Save into：", str(current_face_dir) + "/img_face_" + str(cnt_ss) + ".jpg")
                    else:
                        print("请在按 'S' 之前先按 'N' 来建文件夹 / Please press 'N' before 'S'")

    # 显示人脸数
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 40), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    # 添加说明
    cv2.putText(img_rd, "N: New face folder", (20, 370), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "S: Save current face", (20, 410), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # 按下 'q' 键退出
    if kk == ord('q'):
        break

    cv2.imshow("camera", img_rd)

# 释放摄像头
cap.release()
cv2.destroyAllWindows()