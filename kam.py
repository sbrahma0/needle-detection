import cv2 as cv
import numpy as np
'''
cap = cv2.VideoCapture('E:\\MD\\kamakshi\\2.avi')
count = 1
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    cv2.imwrite('E:\\MD\\kamakshi\\new' + "\\%d.jpg" % count, frame)
    cv2.imwrite('E:\\MD\\kamakshi\\new2' + "\\%d.jpg" % count, gray)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
'''
# cap = cv.VideoCapture('E:\\MD\\kamakshi\\2.avi')
import cv2 as cv
import numpy as np
fourcc = cv.VideoWriter_fourcc('M','J','P','G')
out = cv.VideoWriter('E:\\MD\\kamakshi\\output.avi',fourcc, 20.0, (1296,964))

for i in range(1, 822):

    # cap = cv.imread('pics\\1.jpg')
    cap = cv.imread('E:\\MD\\kamakshi\\new\\' + str(i) + '.jpg')
    h, w = cap.shape[:2]
    print (w)
    #cap = cv.circle(cap, (int((w/2)+112), int((h/2)-35)), 400, (255, 255, 255), 100)
    gray2 = cv.cvtColor(cap, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray2, (9, 9), 0)
    edges = cv.Canny(gray, 120, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, 2 * np.pi / (180), 5, None, 50, 10)
    # print(lines)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv.line(cap, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 3, cv.LINE_AA)

    cv.imshow('cap', cap)
    out.write(cap)
    if cv.waitKey(10) == 27:
        break

cv.destroyAllWindows()  # Destroys all window after pressing 'ESC'