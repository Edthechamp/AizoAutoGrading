import cv2
import numpy as np
from imutils.perspective import four_point_transform
import pytesseract
import re


"""
Add error checking - ie missing number, etc
Create function for extracting the student's code
"""

def extract_document(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imshow('grayscaled image', gray)
        cv2.waitKey(0)

    #normalize illumination
    illum = cv2.GaussianBlur(gray, (101, 101), 0)
    norm = cv2.normalize(gray / (illum + 1e-6), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if debug:
        cv2.imshow('normalized illumination', norm)
        cv2.waitKey(0)

    #detect edges
    blurred = cv2.GaussianBlur(norm, (11, 11), 0)
    edges = cv2.Canny(blurred, 30, 90)
    if debug:
        cv2.imshow('edges after Canny', edges)
        cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel)
    if debug:
        cv2.imshow('edges after dilation', edges)
        cv2.waitKey(0)

    #find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    result = image.copy()
    warped = image.copy()

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            cv2.drawContours(result, [approx], -1, (0, 255, 0), 4)
            warped = four_point_transform(warped, approx.reshape(4, 2))
    if debug:
        cv2.imshow('contoured page', result)
        cv2.waitKey(0)

    if debug:
        cv2.imshow('warped image', warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return warped



def get_topic_boxes(image, debug=False):
    final = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    if debug:
        cv2.imshow('edges of document after canny', edges)
        cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel)
    if debug:
        cv2.imshow('edges of document after dilation', edges)
        cv2.waitKey(0)


    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = image.shape[0] * image.shape[1]

    boxes = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(approx)

        if len(approx) == 4 and img_area * 0.05 < area < img_area * 0.9:
            boxes.append(approx)

    boxes = sorted(boxes, key=cv2.contourArea, reverse=True)
    if debug:
        for box in boxes:
            cv2.drawContours(final, [box], -1, (0, 255, 0), 4)

        cv2.imshow('marked topic boxes', final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return boxes



def get_box_title(image, box, debug=False):
    x, y, w, h = cv2.boundingRect(box)

    title_region = image[y:y + int(h * 0.1), x:x+w]
    if debug:
        cv2.imshow('title of the box', title_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    gray = cv2.cvtColor(title_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    title = pytesseract.image_to_string(thresh, config='--psm 7', lang='lav').strip()

    #clean title
    title = re.sub(r'[^A-ZĀČĒĢĪĶĻŅŠŪŽa-zāčēģīķļņšūž ]', '', title).strip() 

    return title

def get_box_answers(image, box, debug=False):
    x, y, w, h = cv2.boundingRect(box)

    #ignore the title bar
    y += int(0.1 * h)
    h = int(h * 0.9)

    roi = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=20, maxRadius=40)

    if debug:
        copy = image.copy()
        if circles is not None:
            for cx, cy, r in np.round(circles[0]).astype(int):
                cv2.circle(copy, (x+cx, y+cy), r, (0, 255, 0), 2)

        cv2.imshow('marked answer circles', copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #group circles per row
    rows = {}
    for cx, cy, r in np.round(circles[0]).astype(int):
        matched = False
        for row_y in rows:
            if abs(cy - row_y) < 10:
                rows[row_y].append((cx, cy, r))
                matched = True
                break
        if not matched:
            rows[cy] = [(cx, cy, r)]

    #split rows per questions
    #TODO: Create seperate groups for right and left questions, THEN add them so no need for question ocr
    questions = []
    right_questions = []
    left_questions = []
    for row_y, row_circles in sorted(rows.items()):
        x_positions = sorted(set(c[0] for c in row_circles))

        gaps = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)

        if max_gap > avg_gap * 2:
            split_x = x_positions[gaps.index(max_gap)]
            left = [c for c in row_circles if c[0] <= split_x]
            right = [c for c in row_circles if c[0] > split_x]
            left_questions.append(sorted(left, key=lambda c: c[0]))
            right_questions.append(sorted(right, key=lambda c: c[0]))
            continue
        questions.append(sorted(row_circles, key=lambda c: c[0]))
    questions += left_questions + right_questions

    questions_per_row = len(questions[0])
    collapsed = np.array([c for q in questions for c in q])

    r = int(np.median(collapsed[:,2]))
    yy, xx = np.ogrid[-r:r, -r:r]
    circle_mask = (xx**2 + yy**2 <= r**2)
    crops = np.array([gray[cy-r:cy+r, cx-r:cx+r] for cx, cy, _ in collapsed])
    means = crops[:, circle_mask].mean(axis=1)

    means_2d = means.reshape(-1, questions_per_row)
    answers = means_2d.argmin(axis=1)
    answers = np.array(list("ABCD"))[answers]
    result = {str(i+1): str(answers[i]) for i in range(len(answers))}

    return result


def get_student_code(image, debug=False):
    h, _ = image.shape[:2]

    roi = image[int(0.025*h):int(0.2*h), :]
    if debug:
        copy = roi.copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    if debug:
        cv2.imshow('edges before dilation', edges)
        cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel)
    if debug:
        cv2.imshow('edges after dilation', edges)
        cv2.waitKey(0)


    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_area = roi.shape[0] * roi.shape[1]

    boxes = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(approx)

        if len(approx) == 4 and 0.005 * roi_area < area < roi_area * 0.2:
            boxes.append(approx)
            if debug:
                cv2.drawContours(copy, [cnt], -1, (0, 255, 0), 4)

    if debug:
        cv2.imshow('code boxes', copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #sort boxes based on x coordinates
    boxes = sorted(boxes, key=lambda b: cv2.boundingRect(b)[0])

    box_images = []
    for box in boxes:
        x, y, w, h = cv2.boundingRect(box)
        box_images.append(roi[y:y+h, x:x+w])

    ########
    #pass these images to AI to determine what number is written
    ########


def get_answers(image_path, debug=False):
    # edv: uztaisu ka passo path, nevis image, lai nav jaimporto cv2 citos failos
    try:
        image = cv2.imread(image_path)
    except:
        if debug:
            print(f"process_image.get_answers() could not load image from path {image_path}")
        return {"error":"failed reading image"}


    document_img = extract_document(image, debug)
    boxes = get_topic_boxes(document_img, debug)

    result = {}

    for box in boxes:
        title = get_box_title(document_img, box, debug)
        answers = get_box_answers(document_img, box, debug)

        result[title] = answers

    get_student_code(document_img, debug)

    if True:
        print(result)

    return result

get_answers('image.jpg', False)

