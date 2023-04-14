import cv2

frame_width = 640
frame_height = 480

#setup camera dimensions
full_size_frame_width = 1200
full_size_frame_height = 720

#camera
Camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
Camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
Camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
Camera.set(cv2.CAP_PROP_FPS, 30)

#defining the colors
Red = (0, 0, 255)
Orange = (0, 127, 255)
Yellow = (0, 255, 255)
Green = (0, 255, 255)
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

#Window name
cv2.namedWindow('Motion Detection Window')
cv2.setMouseCallback('Motion Detection Window', MouseClick)

RoiCreated = False

#frame management
while True:
    Ignore, Control_Frame = Camera.read()
    Ignore, Test_Frame = Camera.read()
    Control_Frame_Flipped = cv2.flip(Control_Frame, 1)
    Test_Frame_Flipped = cv2.flip(Test_Frame, 1)
    Control_Gray_Frame = cv2.cvtColor(Control_Frame_Flipped, cv2.COLOR_BGR2GRAY)
    Test_Gray_Frame = cv2.cvtColor(Test_Frame_Flipped, cv2.COLOR_BGR2GRAY)
    Differential_Frame = Control_Gray_Frame - Test_Gray_Frame

#coleect mosue clicks roi
    if evt == 1:
        RoiCreated = False
    if evt == 4:
        Roi = Differential_Frame[Down_Y : Up_Y, Down_X : Up_X]
        RoiCreated = True

#detect contours roi
    if RoiCreated == True:
        ret, Threshold_Frame = cv2.threshold(Roi, Contour_Threshold, 255, cv2.THRESH_BINARY)
        Contours_Found, hierarchy = cv2.findContours(Threshold_Frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(Roi,Contours_Found, Solid, Green, Line_Thickness)
        Differential_Contours_Found, hierarchy = cv2.findContours( Roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.rectangle(Control_Frame_Flipped, (Down_X, Down_Y), (Up_X, Up_Y), Green, Line_Thickness)

#detect motion within roi
        for Contour in Differential_Contours_Found:
            Area = cv2.contourArea(Contour)
            if Area > Area_Low and Area < Area_High:
                cv2.putText(Control_Frame_Flipped, Text, Text_Bottom_Left_Origin, Text_Font, Text_Font_Scale, Text_Colour, Text_Thickness, cv2.LINE_AA)
                cv2.rectangle(Control_Frame_Flipped, (Down_X, Down_Y), (Up_X, Up_Y), Red, Line_Thickness)

#display frame
    Full_Size_Control_Frame = cv2.resize(Control_Frame_Flipped, (full_size_frame_width, full_size_frame_height))
    cv2.imshow('Motion Detection Window', Full_Size_Control_Frame)
    cv2.moveWindow('Motion Detection Window', 0, 0)

#press q to quit
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
Camera.release()
