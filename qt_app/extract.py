import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
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

        #Remove shadows
        bg_img = cv2.medianBlur(gray, 51)  # just blur the original, no dilation
        diff_img = cv2.divide(gray, bg_img, scale=230)  # normalize lighting gently
        gray = cv2.addWeighted(diff_img, 0.2, gray, 0.8, 0) 

        scanned_document = {}

        #extract code
        code_area = gray[self.code_box[0][1]:self.code_box[2][1], self.code_box[0][0]:self.code_box[1][0]]
        code_area = cv2.rotate(code_area, cv2.ROTATE_90_CLOCKWISE)
        copy = code_area.copy()

        _, binary = cv2.threshold(code_area, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        blurred = cv2.GaussianBlur(binary, (3, 3), 0)

        edges = cv2.Canny(blurred, 30, 90)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            if len(cnt) < 3:
                continue
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.6 < aspect_ratio < 1.4 and 1000 < area < 8000:
                boxes.append((x, y, w, h))

        # if we got fewer than 6, split the widest box (dash merge)
        boxes = sorted(boxes, key=lambda b: b[0])
        if len(boxes) < 6:
            median_w = np.median([b[2] for b in boxes])
            new_boxes = []
            for b in boxes:
                x, y, w, h = b
                if w > median_w * 1.5:  # this one is merged
                    half = w // 2
                    new_boxes.append((x, y, half, h))
                    new_boxes.append((x + half, y, half, h))
                else:
                    new_boxes.append(b)
            boxes = sorted(new_boxes, key=lambda b: b[0])

        # crop with dynamic margin to remove box border
        def crop_to_digit(code_area, x, y, w, h):
            crop = code_area[y+5:y+h-5, x+5:x+w-5]
            _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = cv2.findNonZero(binary)
            if coords is None:
                return crop
            cx, cy, cw, ch = cv2.boundingRect(coords)
            pad = 4
            return crop[max(0,cy-pad):cy+ch+pad, max(0,cx-pad):cx+cw+pad]

        box_images = []
        for x, y, w, h in boxes:
            digit_img = crop_to_digit(binary, x, y, w, h)
            box_images.append(digit_img)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Extractor.load_model("mnist_model.pth", device)

        code_digits = []
        for img in box_images:  # sort left to right
            #remove box around the number
            coords = cv2.findNonZero(img)
            x, y, w, h = cv2.boundingRect(coords)

            img = img[x:h, y:w]
            img = cv2.bitwise_not(img)
            #cv2.imshow('code boxes', img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            digit, confidence = Extractor.predict(img, model, device)
            print("plis", digit, confidence)
            code_digits.append(str(digit))

        scanned_document['code'] = ''.join(code_digits)

        #extract answers
        for box in self.topic_boxes:
            tl, tr, br, bl = box["pts"]
            box_label = box["label"]

            box_img = gray[tl[1]:bl[1], tl[0]:tr[0]]
            #---------
            #TEMPORARY
            #---------
            box_img = cv2.rotate(box_img, cv2.ROTATE_90_CLOCKWISE)


            box_h = box_img.shape[0]
            box_img = box_img[int(0.1*box_h):box_h, :]
            min_r = 13 # tweak these ratios
            max_r = 25

            kernel = np.ones((3,3), np.uint8)
            gradient = cv2.morphologyEx(box_img, cv2.MORPH_GRADIENT, kernel)

            circles = cv2.HoughCircles(gradient, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=min_r, maxRadius=max_r)

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

    def load_model(model_path, device):
        model = DigitCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    def predict(image_array, model, device):
        from PIL import Image

        scan_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        img = Image.fromarray(image_array)  # numpy array instead of file path
        tensor = scan_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)

        return predicted.item(), confidence.item()




class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))
