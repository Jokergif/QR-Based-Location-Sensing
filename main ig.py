import cv2
from functions import decode_qr
from pyzbar import pyzbar
import numpy as np
from collections import namedtuple


#VARIABLES 
qr_size = 6.2  # centimetres
camera_matrix = np.array([
        [612.21443697, 0, 333.76740695],
        [0, 611.27364813, 241.82057016],
        [0, 0, 1]
        ])

#display text(distance to QR)
text_size = 1
text_thickness = 2
image_resolution = (640, 480)
run_onetime = 0 #default 0 for continuous running
press_s_to_run = 0 #default 1 for continuous running

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
    Point = namedtuple('Point', ['x', 'y'])




    while 1 :#or 0xFF == ord('s') or press_s_to_run:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        cv2.circle(frame, (320,240), 5, (0,0,255), -1)
        
        # Decode the QR code in the frame
        qr_data = decode_qr(frame,qr_size,camera_matrix)
        
        # Display the frame with the QR code highlighted
        cv2.imshow("QR Code Scanner", frame)
        
        # Break the loop if a QR code is detected
        if qr_data:
            print(f"QR Code Data: {qr_data}")
            if run_onetime: break

    # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
