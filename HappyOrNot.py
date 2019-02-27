import cv2
import numpy as np
from keras.models import load_model
from statistics import mode


#情绪标签
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'} 

frame_window = 10 #调控情绪更新反应
emotion_window = []

emotion_offsets = (20, 40) #表情检测块偏差

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') #cv2载入人脸识别模型
emotion_classifier = load_model('./emotion_model.hdf5') #keras载入情绪分类模型

emotion_target_size = emotion_classifier.input_shape[1:3] #表情分类器表情检测大小 64*64

# 打开摄像头捕捉
cv2.namedWindow('window_frame')
cap = cv2.VideoCapture(0) #0是自带摄像头

#应用偏差
def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

while cap.isOpened():
    ret, bgr_image = cap.read() #读视频帧

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) #转灰度图，用于人脸检测，可加快检测速度
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) #转RGB图 ，cv2默认为BGR顺序

    #使用opencv检测出所有的人脸
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets) #扩大人脸区域偏差
        gray_face = gray_image[y1:y2, x1:x2] #获得灰度图人脸部分
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size)) #获取的人脸灰度图大小调整和情绪分类模型大小一样
        except:
            continue

        # 灰度图像预处理
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        emotion_prediction = emotion_classifier.predict(gray_face) #对图像进行分类预测
        #print(emotion_prediction)
        #混淆矩阵的处理
        emotion_probability = np.max(emotion_prediction)
        #print(emotion_probability)
        emotion_label_arg = np.argmax(emotion_prediction) #获取索引（情绪）

        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text) #添加入emotion_window列表

        if len(emotion_window) > frame_window: #如果获得的情绪列表大于10 ，就删掉第一个情绪，更新新情绪
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window) #从离散的一组情绪里选取最多出现的一个
        except:
            continue

        #根据情绪的可能性 取该情绪颜色的明暗，可能性低--》暗，转化为数组
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int) #将数组里的值取整数
        color = color.tolist() #array转化为list

        #绘人脸位置框和表情文字
        x, y, w, h = face_coordinates
        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), color, 2)

        x, y = face_coordinates[:2]
        cv2.putText(rgb_image, emotion_mode, (x + 0, y -10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 1, cv2.LINE_AA)

    #显示到窗口
    bgr_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    
    #cv2结束指令
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#释放资源 关闭窗口
cap.release()
cv2.destroyAllWindows()