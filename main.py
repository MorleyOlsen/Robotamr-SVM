import threading

import cv2

from img_rec import img_rec
from robotic_arm.my_serial import MySerial

# 单片机功能选择
SINGLE_CHIP_FUNCTION_DATA = "3001070155a1ff"
# 控制机械臂位置
CONTROL_ROBOTIC_ARM_POSITION_DATA = "3001070155a11131"
# 控制机械臂抓取
CONTROL_ROBOTIC_ARM_GRAB_DATA = "3002070155a131"

vs = cv2.VideoCapture(0)
image_processing = img_rec.ImageProcessing()
# 创建串口对象
my_serial = MySerial('/dev/ttyS4', baudrate=115200, timeout=1)
# 创建串口接收远程
t_serial = threading.Thread(target=my_serial.receive_msg)
t_serial.start()
current_state = 0  # 当前状态
next_state = 0  # 下一个状态
is_grab = False  # 判断是否抓取
source_location = 0  # 最终搬运的源位置

# my_serial.send_msg(SINGLE_CHIP_FUNCTION_DATA + "31")
while True:
    current_state = next_state
    if current_state == 0:
        my_serial.send_msg(CONTROL_ROBOTIC_ARM_POSITION_DATA)
        print("控制机械臂移动。")
        next_state = 1
    elif current_state == 1:
        if my_serial.recv_msg[12:16] == "2131":
            print("机械臂已到达仓库1。")
            next_state = 2
            my_serial.recv_msg = ""
    elif current_state == 2:
        print("开始拍摄照片。")
        for i in range(30):
            ret, frame = vs.read()
        cv2.imwrite("./pic.jpg", frame)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
        print("拍摄照片完成，照片保存在当前目录下的pic.jpg，开始识别。")
        image_thresh, cargo_location = image_processing.image_position(frame)
        cargo_location_sort = image_processing.image_sort(cargo_location)
        rec_result = image_processing.image_recognize(cargo_location, cargo_location_sort, frame)
        print("识别结束，结果为{}".format(rec_result))
        if is_grab and rec_result != {}:
            # sorted 可以对所有可迭代的对象进行排序操作。
            # sorted(iterable, key=None, reverse=False)
            # iterable -- 可迭代对象。
            # key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
            # reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
            # 返回值 -- 返回重新排序的列表。
            list_sort = sorted(rec_result.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
            print("排序结束，结果为{}".format(list_sort))
            source_location = list_sort[0][0] + 1
            next_state = 3
        else:
            break
    elif current_state == 3:
        my_serial.send_msg(CONTROL_ROBOTIC_ARM_GRAB_DATA + "1{}21".format(source_location))
        next_state = 4
    elif current_state == 4:
        if my_serial.recv_msg[12:16] == "4131":
            print("机械臂已搬运完毕。")
            break
        break
    else:
        break

my_serial.THREAD_CONTROL = False
