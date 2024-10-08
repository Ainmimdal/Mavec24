#!/usr/bin/env python3
import cv2
import numpy as np

def main(video_source):
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 13), 0)
        edge = cv2.Canny(blur, 50, 55)

        height, width = edge.shape
        i = 30
        j = 120
        k = 0

        poly = np.array([
            [
                (0, height - k),
                (int(width / 2) - i, int(height / 2) + j),
                (int(width / 2) + i, int(height / 2) + j),
                (width, height - k)
            ]
        ])

        upPoly = np.array([
            [
                (10, height - 200),
                (int(width / 2) - 1, int(height / 2) + 20),
                (int(width / 2) + 1, int(height / 2) + 20),
                (width, height - 200)
            ]
        ])

        hoodPoly = np.array([[
            (360, width),
            (360, 670),
            (440, 670),
            (440, width)
        ]])

        blank = np.zeros_like(edge)
        mask = blank.copy()
        mask = cv2.fillPoly(mask, pts=[poly], color=255)

        hoodMask = blank.copy()
        hoodMask = cv2.fillPoly(hoodMask, pts=[hoodPoly], color=255)
        hoodMask = cv2.bitwise_and(hoodMask, mask)
        mask = cv2.bitwise_xor(mask, hoodMask)

        maskedImg = cv2.bitwise_and(edge, mask)
        upMask = blank.copy()
        upMask = cv2.fillPoly(upMask, pts=[upPoly], color=255)
        upperMaskImg = cv2.bitwise_and(edge, upMask)

        upperLines = cv2.HoughLinesP(upperMaskImg, rho=3, theta=np.pi/45, threshold=20, lines=np.array([]), minLineLength=15, maxLineGap=5)
        lines = cv2.HoughLinesP(maskedImg, rho=3, theta=np.pi/45, threshold=20, lines=np.array([]), minLineLength=30, maxLineGap=5)

        final = image.copy()
        rawLinesImage = image.copy()
        color = (0, 255, 0)

        speed = 7.46
        if lines is not None:
            speed = 3
            if len(lines) == 1:
                lines = [lines]

            avglines, error = avg_line(blank, lines)
            if abs(error) > 1:
                speed = 2

            final = draw_lines(final, avglines, color)
            rawLinesImage = draw_lines(rawLinesImage, lines, (255, 0, 0))
        else:
            if upperLines is not None:
                avgULines, Uerror = avg_line(blank, upperLines)
                if abs(Uerror) > 0.5:
                    speed = 5
                    print("!------- speed reduced ; uplines_error == ", Uerror, "-------!")

            error = 0

        steering = translate(-1.5, 1.5, -4.5, 4.5, float(error))

        # You can use speed and steering to control the car using GPIO or other methods

        cv2.imshow("CANNY IMAGE", edge)
        cv2.imshow("IMAGE LINES", rawLinesImage)
        cv2.imshow("FINAL", final)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def translate(inp_min, inp_max, out_min, out_max, inp_error):
    return (float(inp_error - inp_min) * float(out_max - out_min) / float(inp_max - inp_min) + float(out_min))

def draw_lines(image, lines, color):
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(image, (x1, y1), (x2, y2), color, 10)
    return image

def draw_texts(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 3.0
    color = (0, 0, 255)
    thickness = 2
    return cv2.putText(image, text, (0, 400), font, fontScale, color, thickness)

def make_coordinates(image, line):
    if isinstance(line, np.ndarray):
        slope, intercept = line
        y1 = image.shape[0]
        y2 = int(y1 * (3/5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    else:
        return np.array([0, 0, 0, 0])

def avg_line(image, lines):
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope <= 0:
            if slope != 0:
                left_lines.append((slope, intercept))
                left_lines.append((slope, intercept))
                left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    left_line_avg = np.average(left_lines, axis=0)
    right_line_avg = np.average(right_lines, axis=0)

    left_line = make_coordinates(image, left_line_avg)
    right_line = make_coordinates(image, right_line_avg)

    left_line_avg_fix = left_line_avg if isinstance(left_line_avg, np.ndarray) else np.array([0, 0])
    right_line_avg_fix = right_line_avg if isinstance(right_line_avg, np.ndarray) else np.array([0, 0])

    error = (left_line_avg_fix[0] + right_line_avg_fix[0]) / 2

    return np.array([left_line, right_line]), error

if __name__ == "__main__":
    # Change video_source to 0 for USB camera or provide the path to a video file
    #video_source = 0  # Uncomment this line for USB camera
    video_source = "/home/mavecpi/Downloads/testvid.mp4"  # Uncomment this line for video file

    main(video_source)