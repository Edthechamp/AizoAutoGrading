import cv2
from imutils.perspective import four_point_transform
import numpy as np

class Extractor():
    def __init__(self, camera_feed, document_pts, topic_boxes, code_box):
        super().__init__()
        self.camera = camera_feed
        self.documents_pts = document_pts
        self.topic_boxes = topic_boxes
        self.code_box = code_box
    
    def scan_answers(self):
        frame = self.camera.latest_frame

        document_img = four_point_transform(frame, self.documents_pts)
        gray = cv2.cvtColor(document_img, cv2.COLOR_BGR2GRAY)

        scanned_document = {}

        #extract code
        #---------
        #TEMPORARY
        #---------
        scanned_document['code'] = '123456'

        #CURRENTLY NEEDS HEAVY TWEAKING

        #extract answers
        for box in self.topic_boxes:
            tl, tr, br, bl = box["pts"]
            box_label = box["label"]

            box_img = gray[tl[1]:bl[1], tl[0]:tr[0]-int(0.1 * (tr[0] - tl[0]))]
            #---------
            #TEMPORARY
            #---------
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_COUNTERCLOCKWISE)


            box_h = box_img.shape[0]
            min_r = max(5, int(box_h * 0.025)) - 5 # tweak these ratios
            max_r = max(10, int(box_h * 0.045))

            circles = cv2.HoughCircles(box_img, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=23, minRadius=min_r, maxRadius=max_r)

            copy = box_img.copy()
            if circles is not None:
                for cx, cy, r in np.round(circles[0]).astype(int):
                    cv2.circle(copy, (cx, cy), r, (0, 255, 0), 2)

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
            crops = np.array([box_img[cy-r:cy+r, cx-r:cx+r] for cx, cy, _ in collapsed])
            means = crops[:, circle_mask].mean(axis=1)

            means_2d = means.reshape(-1, questions_per_row)
            answers = means_2d.argmin(axis=1)
            answers = np.array(list("ABCD"))[answers]
            result = {str(i+1): str(answers[i]) for i in range(len(answers))}

            scanned_document[box_label] = result
        
        print(scanned_document)
    
        return scanned_document




