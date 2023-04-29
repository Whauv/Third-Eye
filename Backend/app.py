from flask import Flask, render_template, Response
#from imutils.video import VideoStream
import cv2

app = Flask(__name__)
frame_width = 640
frame_height = 480
#vs = VideoStream(src=0).start()
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
capture.set(cv2.CAP_PROP_FPS, 30)

#setup camera dimensions
full_size_frame_width = 640
full_size_frame_height = 480

thres = 0.5

#defining the colors
Red = (0, 0, 255)
Orange = (0, 127, 255)
Yellow = (0, 255, 255)
Green = (0, 255, 0)
Blue = (255, 0, 0)
Indigo = (128, 0, 55)
Voilet = (250, 0, 175)

#global variables
Contour_Threshold = 1
Contour_Colour = Orange
Contour_Line_Thickness = 1
Line_Thickness = 3
Area_Low = 25
Area_High = frame_width * frame_height
Solid = -1

#global text variables
Text = "Warning!!!"
Text_Colour = Red
Text_Bottom_Left_Origin = (int(frame_width / 20), int(frame_height / 6))
Text_Font = cv2.FONT_HERSHEY_SIMPLEX
Text_Font_Scale = 2
Text_Thickness = 2

#Mouse clicks
evt = 0
Down_X = 0
Down_Y = 0
Up_X = 0
Up_Y = 0
def MouseClick(event, xPos, yPos, flags, parameters):
    global evt
    global Down_X
    global Down_Y
    global Up_X
    global Up_Y
    if event == cv2.EVENT_LBUTTONDOWN:
        Down_X = int(xPos * 0.5)
        Down_Y = int(yPos * 0.5)
        evt = event
    if event == cv2.EVENT_LBUTTONUP:
        Up_X = int(xPos * 0.5)
        Up_Y = int(yPos * 0.5)
        evt = event

#cv2.namedWindow('Third Eye')
# cv2.setMouseCallback('Full_Size_Control_Frame', MouseClick)

# RoiCreated = False

classNames = []
classFile = 'coco.names'
with open(classFile,'r') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightspath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightspath, configpath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


@app.route('/')
def home():
    return render_template('index.html')

def generate_frames():
    RoiCreated = False
    while True:
        Ignore, Control_Frame = capture.read()
        Ignore, Test_Frame = capture.read()
        Control_Gray_Frame = cv2.cvtColor(Control_Frame, cv2.COLOR_BGR2GRAY)
        Test_Gray_Frame = cv2.cvtColor(Test_Frame, cv2.COLOR_BGR2GRAY)
        Differential_Frame = Control_Gray_Frame - Test_Gray_Frame

        #collect mouse clicks roi
        if evt == 1:
            RoiCreated = False
        if evt == 4:
            Roi = Differential_Frame[Down_Y : Up_Y, Down_X : Up_X]
            RoiCreated = True

        classIds, confs, bbox = net.detect(Control_Frame,confThreshold = thres)
        print(classIds,bbox)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
                cv2.rectangle(Control_Frame,box,color = Green,thickness = 2)
                cv2.putText(Control_Frame,(classNames[classId - 1].upper()),(box[0] + 10,box[1]+ 30),Text_Font,1,Green,2)
                cv2.putText(Control_Frame,(str(round(confidence*100,2))),(box[0] + 200,box[1]+ 30),Text_Font,1,Green,2)
        
        # if RoiCreated == True:
        #     ret, Threshold_Frame = cv2.threshold(Roi, Contour_Threshold, 255, cv2.THRESH_BINARY)
        #     Contours_Found, hierarchy = cv2.findContours(Threshold_Frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     cv2.drawContours(Roi,Contours_Found, Solid, Green, Line_Thickness)
        #     Differential_Contours_Found, hierarchy = cv2.findContours( Roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     cv2.rectangle(Control_Frame, (Down_X, Down_Y), (Up_X, Up_Y), Green, Line_Thickness)
        #     for Contour in Differential_Contours_Found:
        #         Area = cv2.contourArea(Contour)
        #         if Area > Area_Low and Area < Area_High:
        #            cv2.putText(Control_Frame, Text, Text_Bottom_Left_Origin, Text_Font, Text_Font_Scale, Text_Colour, Text_Thickness, cv2.LINE_AA)
        #            cv2.rectangle(Control_Frame, (Down_X, Down_Y), (Up_X, Up_Y), Red, Line_Thickness)

        #     Full_Size_Control_Frame = cv2.resize(Control_Frame, (full_size_frame_width, full_size_frame_height))
        #     cv2.imshow('Motion Detection Window', Full_Size_Control_Frame)
        #     cv2.moveWindow('Motion Detection Window', 0, 0)
            ret, buffer = cv2.imencode('.jpg', Control_Frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: Control_Frame/jpeg\r\n\r\n' + frame + b'\r\n')

def video():
    Control_Frame = capture.read()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/normal_video')
def normal_video():
    return Response(video(), mimetype='multipart/x-mixed-replace; boundary=frame')
#@app.route('/toggle')
#while generate_frames() == True:
    #if cv2.waitKey(1) & 0xff == ord('toggle'):
      #  break
   # else:
       # continue

if __name__ == '__main__':
    app.run(debug=True)


