"""
To do IP camera output and image capture.
"""
import argparse
import time
import datetime
import cv2

#ADD is the address to access IP camera.
ADD = 'rtsp://admin:1234@192.168.100.253:554/1/1'
# Cap can call protocols to RTSP and stream videos from cameras.
cap = cv2.VideoCapture(ADD)
state = ''


# 프레임에 검은색 픽셀 사각형 그리기.    
def find_black_pixels(frame, th,y_value):

    """ Find the black pixels with threshold and draw a rectangle.
        
        Args:
            frame (frame): Original image for graying
            th (int): Threshold for gray value conversion
            y_value (int): Range to apply the rectangle on the y-axis
            
        Return:
            frame: The selected threshold part is overlaid on the image.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,30,2)
    
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        #contour 영역 중에 사각형을 씌울 조건걸기
        if cv2.contourArea(contour) > 600 or y >= y_value:
            continue
        # 사각형 및 텍스트 표시하기
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        pixel = f"({x}, {y})"
        cv2.putText(frame, pixel, (x+3, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))



def run_ipcam(args: argparse.Namespace):
    """It has one parameter, 'timer' is the number that sets the capture time period."""
    
    """ Print and save the image and returns the status of the program.
        
        Args:
            args.m (int) : time interval for saving pictures (minutes)
            args.t (int) : Argument value to go to 'find_black_pixels' method
            args.y (int) : Argument value to go to 'find_black_pixels' method
            
        Returns:
            String : Pressing the keyboard 'q' returns the character 'end'.
    """

    try:
        ret, frame = cap.read()
        find_black_pixels(frame,args.t,args.y)
        video = cv2.resize(frame, dsize=(1920,1024))        
        cv2.imshow("IP Camera stream", video)

        secs = time.time()

        if datetime.datetime.now().minute % args.m == 0 and datetime.datetime.now().second == 0:
            """Capture the video you are streaming and save it as an image."""

            epoch = int(secs)

            print('Saved frame number : ' + str(int(cap.get(1))))
            cv2.imwrite("images/%d.png" % epoch, frame)

            tm = time.gmtime(secs)
            print("[{0}년{1}월{2}일 {3}시 {4}분] Saved frame - {5}".format(tm.tm_year, tm.tm_mon,
                                                tm.tm_mday, tm.tm_hour + 9, tm.tm_min, epoch))

            time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            global state
            state = 'end'
            print('프로그램 종료')
            
        
    except Exception as E:
        print(E)
   
    return state

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="image save and rectangle on black pixel")
    parser.add_argument("-m", default=1, type=int, help="time interval for saving pictures (minutes) ")
    parser.add_argument("-t", default=60, type=int, help="cv threshold value")
    parser.add_argument("-y", default=1000, type=int, help="hide y axis")
    args = parser.parse_args()
    while state != 'end':
       run_ipcam(args)