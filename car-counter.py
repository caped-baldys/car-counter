from ultralytics import YOLO
import cv2
import cvzone  # To display the detections (optional)
import math
from sort import *
#cap = cv2.VideoCapture(0)  # 0 - 1 index of the video capture in device
cap = cv2.VideoCapture("traffic_cctv.mp4")
cap.set(3, 1280)  # sets the width of image viewer or capture commonly set to 640
cap.set(4, 760)  # sets the heigth of the image viewer box or capture ... set to 480

model = YOLO("YOLO-Weights/yolov8n.pt")
names = model.names

mask = cv2.imread("mask.png")
#initialize tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits = [360,400,1170,400]
total_car = [] 

size = (1280, 720) 
result_vid = cv2.VideoWriter('result.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                             10, size)

while True:
    success, img = cap.read()
    # Apply the binary mask to isolate the region of interest
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            #bounding box
            x1,y1,x2,y2 = box.xyxy[0] #gets the coordinates of the box in the image 
        
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2-x1, y2-y1

            #confidence
            conf = math.ceil((box.conf[0]*100))/100
        
            #class name of object
            cls = int(box.cls[0])
            name = names[cls]
            
            if name == "car" or name == "bus" or name == "motorbike" or name == "truck" and conf >0.3:
                # cvzone.putTextRect(img,f'{name}{conf}',(max(2,x1),max(35,y1)),scale=0.7,thickness=1,offset=3)
                # cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt = 5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections =  np.vstack((detections,currentArray))


    resultsTracker = tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5 )

    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2,id = int(x1),int(y1),int(x2),int(y2),int(id)
        w, h = x2-x1, y2-y1
        
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,255))
        cvzone.putTextRect(img,f'{id}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1,offset=3)
        cx ,cy = x1+w//2 , y1+h//2
        cv2.circle(img,(cx,cy),4,(0,255,255),cv2.FILLED)
    
        if limits[0]<cx<limits[2] and (limits[1]-10) <cy <(limits[3]+10):
            if id not in total_car:
                total_car.append(id)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5 )
    cv2.putText(img,f'Count:{len(total_car)}',(50,50),cv2.FONT_HERSHEY_PLAIN,3,(50,50,255),8)
    # Write the frame into the 
    # file 'filename.avi' 
    result_vid.write(img) 
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('s'): 
        break
# write objects 
cap.release() 
result_vid.release()

# Closes all the frames 
cv2.destroyAllWindows() 
   
print("The video was successfully saved") 