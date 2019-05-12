#모듈 호출------------------------------------------------------------------------------
import cv2
import numpy as np
import time 
import dlib
import pybind11

s_img = cv2.imread('test2.png',-1)##############::::

#많이 씀-------------------------------------------------------------------------------

#얼굴 그릴때 씀-------------------------------------------------------------------------
def drawPolyline(im,im2, landmarks, start, end, isClosed=False):

  points = []

  for i in range(start, end+1):
      point = [landmarks.part(i).x, landmarks.part(i).y]
      points.append(point)
  points = np.array(points, dtype=np.int32)
  
  cv2.polylines(im2, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)
#얼굴 검출 부위-------------------------------------------------------------------------
def renderFace(im,im2, landmarks):
    assert(landmarks.num_parts == 68)
    drawPolyline(im,im2, landmarks, 0, 16)           # Jaw line
    drawPolyline(im,im2, landmarks, 17, 21)          # Left eyebrow
    drawPolyline(im,im2, landmarks, 22, 26)          # Right eyebrow
    drawPolyline(im,im2, landmarks, 27, 30)          # Nose bridge
    drawPolyline(im,im2, landmarks, 30, 35, True)    # Lower nose
    drawPolyline(im,im2, landmarks, 36, 41, True)    # Left eye
    drawPolyline(im,im2, landmarks, 42, 47, True)    # Right Eye
    drawPolyline(im,im2, landmarks, 48, 59, True)    # Outer lip
    drawPolyline(im,im2, landmarks, 60, 67, True)    # Inner lip
def renderFace2(im, landmarks, color=(0, 255, 0), radius=3):
  for p in landmarks.parts():
      cv2.circle(im, (p.x, p.y), radius, color, -1)


#얼굴 랜드마크 검출, 마스크용으로 쉽게하기위해 따로 호출-----------------------------------

faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#Prev
prev_face = None
prev_idx = 0
PREV_MAX = 100
global x
global y
global cnt

#마스크 이미지--------------------------------------------------------------------------
mask__ = cv2.imread('test2.png')
mask_h, mask_w, _ = mask__.shape
mask_x, mask_y = mask_w / 2, mask_h / 2

#캠------------------------------------------------------------------------------------
cam = cv2.VideoCapture(0)
cam.set(3,960)
cam.set(4,480)

#얼굴 테두리 뒷 배경 마스크--------------------------------------------------------------
lower_mask = np.array([0,0,0])
#--------------------------------------------------------------------------------------

Rect = 0

cam.read()
time.sleep(0.5)
while 1:


    frame, img = cam.read()
    img2 = cv2.imread('test2.png')
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#--------------------------------------------------------------------------------------
    mask_fi = img_gray
    

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#얼굴 테두리 뒷 배경 마스크--------------------------------------------------------------

    mask = cv2.inRange(hsv,lower_mask,lower_mask)
#얼굴 테두리----------------------------------------------------------------------------
    res = cv2.bitwise_and(img,img, mask= mask)

    faceRects = faceDetector(img, 0)

    landmarksAll = []

    for i in range(0, len(faceRects)):

        Rect = dlib.rectangle(int(faceRects[i].left()),int(faceRects[i].top()),
        int(faceRects[i].right()),int(faceRects[i].bottom()))
    #########################################
    faces = detector(img_gray)
    for face in faces:
        landmarks = landmarkDetector(img,Rect)
       
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y=  landmarks.part(n).y
            landmarks_points.append((x, y))
            landmarksAll.append(landmarks)
    points = np.array(landmarks_points, np.int32)
    
    convexhull = cv2.convexHull(points)
    renderFace(img,res, landmarks)
        #print(convexhull)
   

    cv2.fillConvexPoly(mask_fi,convexhull,255)
    
    #######################################################
        
    
    

    res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(res_gray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cnt = contours[0]
    hull = cv2.convexHull(cnt)
    (x,y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    x_offset = int(x)
    y_offset = int(y)


    
    y1, y2 = int(y_offset-(s_img.shape[0]/2)), int(y_offset + (s_img.shape[0]/2))
    x1, x2 = int(x_offset-(s_img.shape[1]/2)), int(x_offset + (s_img.shape[1]/2))

    alpha_s = s_img[:,:,3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                              alpha_l * img[y1:y2, x1:x2, c])

    cv2.rectangle(res,(x1,y1),(x2,y2),(0,255,255),3)
    #print(center)
    radius = int(radius)
    res = cv2.circle(res, center, radius,(0,0,255),3) # yellow

    cv2.drawContours(res, [hull], 0,(0,255,0), 3)

    cv2.imshow("detect", img)
    cv2.imshow("res",res)
   
    cv2.imshow("Mask", mask_fi)
    cv2.imshow("img2",img2)
    
    k= cv2.waitKey(5) & 0xff
    if k==27:
        cam.release()
        cv2.destroyAllWindows()


        break

cam.release()


cv2.destroyAllWindows()
