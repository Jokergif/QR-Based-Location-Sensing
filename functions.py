import cv2
import numpy as np
from collections import namedtuple
from pyzbar import pyzbar
import ast
Point = namedtuple('Point', 'x y')

# THIS IS WORKING PART THAT YOU COMMENTED FOR TESTING
def decode_qr(frame, qr_size, camera_matrix):
    # Decode the QR codes in the frame
    decoded_objects = pyzbar.decode(frame)
    for obj in decoded_objects:

        points = obj.polygon
       # print(points)
        if len(points) > 4: 
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points
       # [print(tuple(center)) for center in points]
        [cv2.circle(frame, center, 5, (255,0,0),2) for center in points]
        
        n = len(hull)

        #calculate_distance_to_plane(points, qr_size, camera_matrix)
        x_perp, y_paral = get_qr_code_distance(camera_matrix,qr_size,np.array([point for point in points], dtype=np.float32),frame)
        #square on QR
        for j in range(0, n):    
            cv2.line(frame, hull[j], hull[(j + 1) % n], (0, 255, 0), 1)

        # Print the QR code data   
        qr_data = obj.data.decode("utf-8")
        try:
            qr_list = ast.literal_eval(qr_data)
            angle = qr_list[1]*np.pi/180 #converting to radian
        except:
             print("DId not get qr_list format")

        '''
        qr_coord = qr_list[0]
        qr_angle = qr_list[1]
        '''       # print(f"Decoded QR code data: {qr_data}")
        conversion(x_perp,y_paral,qr_list[0][0], qr_list[0][1],angle)
        return qr_data

    return None


    
def get_qr_code_distance(camera_matrix, qr_code_size, points,frame):
    #if 1:
        # Order points to match the real world coordinates
        # extract nested array
        points = [Point(x, y) for x, y in points]
        image_points = np.array([[point.x, point.y] for point in points], dtype=np.float32)
        # Define the real world coordinates of the QR code corners
        
        half_size = qr_code_size / 2
    
    # Define the real world coordinates of the QR code corners with the center at (0, 0, 0)
        real_world_points = np.array([[-half_size, -half_size, 0],
                                  [ half_size, -half_size, 0],
                                  [ half_size,  half_size, 0],
                                  [-half_size,  half_size, 0]], dtype=np.float32)
        

        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1))

        # Solve for pose
        success, rvec, tvec = cv2.solvePnP(real_world_points, image_points, camera_matrix, dist_coeffs)

        if success:
            # The translation vector tvec gives us the position of the QR code relative to the camera
            distance = np.linalg.norm(tvec)
            print("actual distance: ", distance)
            #return distance
            cv2.putText(frame, str(distance), (100,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        # Calculate the perpendicular distance (z-component of the translation vector)
            perpendicular_distance = tvec[2][0]
        
        # Calculate the parallel distance (sqrt of x and y components squared)
            parallel_distance = np.sqrt(tvec[0][0]**2 + tvec[1][0]**2)
            if tvec[0][0] < 0:
                parallel_distance = -parallel_distance
            coord = (perpendicular_distance, parallel_distance)
            cv2.putText(frame, str(coord), (1,300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            #print("perpenciduclar: ", perpendicular_distance)
            #print("parallel: " , parallel_distance)
            return perpendicular_distance,parallel_distance

def conversion(x,y,qr_x,qr_y, angle):
            new_x = x*np.cos(angle) - y*np.sin(angle)
            new_y = x*np.sin(angle) + y*np.cos(angle)

            camera_x = qr_x + new_x
            camera_y = qr_y + new_y

            print("(",camera_x,",", camera_y,")")
            return(camera_x,camera_y)
            """
        else:
            print("Could not solve for the pose of the QR code.")
            return None"""
'''else:
        print("QR code not detected.")
        return None
'''