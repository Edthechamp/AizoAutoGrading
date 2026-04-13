import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import cv2
from imutils.perspective import four_point_transform
import numpy as np


class Extractor():
    def __init__(self, camera_feed, document_pts, topic_boxes, code_box, camera_rotation=0):
        super().__init__()
        self.camera = camera_feed
        self.documents_pts = document_pts
        self.topic_boxes = topic_boxes
        self.code_box = code_box
        self.camera_rotation = camera_rotation
    
    def scan_answers(self):
        frame = self.camera.latest_frame

        if self.camera_rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.camera_rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.camera_rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        document_img = four_point_transform(frame, np.array(self.documents_pts))
        gray = cv2.cvtColor(document_img, cv2.COLOR_BGR2GRAY)

        #Remove shadows
        bg_img = cv2.medianBlur(gray, 51)  # just blur the original, no dilation
        diff_img = cv2.divide(gray, bg_img, scale=230)  # normalize lighting gently
        gray = cv2.addWeighted(diff_img, 0.2, gray, 0.8, 0) 

        scanned_document = {}

        #extract code
        cx, cy, cw, ch = self.code_box
        code_area = gray[cy:cy+ch, cx:cx+cw]

        scanned_code = self.get_code(code_area)
        if len(scanned_code) == 6:
            scanned_document['code'] = scanned_code
        else:
            print("error detecting code")
            return

        #extract answers
        for box in self.topic_boxes:
            bx, by, bw, bh = box["box"]
            box_label = box["label"]

            box_img = gray[by:by+bh, bx:bx+bw]
            result = self.detect_answers(box_img)
            scanned_document[box_label] = result
        
        print(scanned_document)
    
        return scanned_document


    
    def get_code(self, gray):
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        #debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


        # Detect boxes in img

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        # Opening = erode then dilate
        # Erode kills anything shorter than the kernel, dilate restores what survived
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)


        # Because kernel is thin, it detects multiple lines per actual box line, making it appear thick.
        # This makes each detected vertical line only 1 pixel wide
        contours, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        v_lines_1px = np.zeros_like(v_lines)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + (w // 2)
            cv2.line(v_lines_1px, (center_x, y), (center_x, y + h), 255, 1)

        v_lines = v_lines_1px




        #cv2.imshow('vertical only', v_lines) 
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        col_proj = np.sum(v_lines, axis=0)

        from scipy.ndimage import uniform_filter1d
        from scipy.signal import find_peaks

        col_proj_smooth = uniform_filter1d(col_proj.astype(float), size=3)

        col_peaks, _ = find_peaks(
            col_proj_smooth,
            height=np.max(col_proj_smooth) * 0.8,
            distance=3
        )

    
        #import matplotlib.pyplot as plt

        #plt.figure(figsize=(15, 4))
        #plt.plot(col_proj_smooth)
        #plt.scatter(col_peaks, col_proj_smooth[col_peaks], color='green', zorder=5, label='detected peaks')
        #plt.axhline(np.max(col_proj_smooth) * 0.8, color='red', linestyle='--', label='height threshold')
        #plt.xlabel('column index')
        #plt.legend()
        #plt.show()

        boxes_x = [(col_peaks[i], col_peaks[i+1]) for i in range(0, len(col_peaks) - 1, 2)]

        if len(boxes_x) % 2 != 0:
            print("Error detecting box lines: not an even number!")
            return
    
        boxes = []

        for (x_left, x_right) in boxes_x:
            strip = h_lines[:, x_left:x_right]
            strip_proj = np.sum(strip, axis=1)

            strip_proj_smooth = uniform_filter1d(strip_proj.astype(float), size=3)
            row_peaks, _ = find_peaks(
                strip_proj_smooth,
                height=np.max(strip_proj_smooth) * 0.3,
                distance=20)

            if len(row_peaks) < 2:
                print(f"Warning: only found {len(row_peaks)} row peaks for strip x={x_left}..{x_right}, skipping")
                continue

            row_peaks = np.sort(row_peaks[np.argsort(strip_proj_smooth[row_peaks])[-2:]])


            y_top, y_bot = row_peaks[0], row_peaks[-1]

            boxes.append((x_left, y_top, x_right-x_left, y_bot-y_top))
    
            #cv2.rectangle(debug, (x_left, y_top), (x_right, y_bot), (0, 255, 0), 3)


        #Detect handwritten digits

        h, w = gray.shape

        box_images = []
        for box in boxes:
            y_pad = 10
            x_pad = 5
            box_img = binary[max(box[1] - y_pad, 0): min(box[1] + box[3] + y_pad, h), max(box[0] - x_pad, 0):min(box[0] + box[2] + x_pad, w)]
            box_img_gray = gray[max(box[1] - y_pad, 0): min(box[1] + box[3] + y_pad, h), max(box[0] - x_pad, 0):min(box[0] + box[2] + x_pad, w)]

            contours, _ = cv2.findContours(box_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mask = np.zeros(box_img.shape, dtype=np.uint8)

            for cnt in contours:
                cv2.drawContours(mask, [cnt], -1, 255, 4)

            box_img_gray[mask == 255] = 255

            by, bx = box_img_gray.shape
            box_img_gray = box_img_gray[y_pad:by - y_pad, x_pad: bx - x_pad]

            box_final = cv2.bitwise_not(box_img_gray) #lets see if this works
            box_images.append(box_final)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Extractor.load_model("emnist_trained.pth", device)

        code_digits = []
        for img in box_images:  # sort left to right
            digit, confidence = Extractor.predict(img, model, device)
            print("plis", digit, confidence)
            code_digits.append(str(digit))

        return ''.join(code_digits)


        
    def detect_circle_bands(self, peaks, v_proj):
        i = 0
        bands = []

        while i < len(peaks) - 1:
            start = peaks[i]
            end = peaks[i+1]

            local_thresh = 0.2 * (v_proj[start] + v_proj[end]) / 2

            is_band = True
            for j in range(start, end):
                if v_proj[j] < local_thresh:
                    is_band = False
                    break

            if is_band:
                bands.append((start, end))
                i += 2 #end cannot be the start of the next peak 
            else:
                i += 1

        # find average width, if some width is signifantly lower, it means it's probably not the circle band
        bands = np.array(bands)
        if len(bands) == 0:
            print("No bands found")
            return bands

        widths = np.diff(bands, axis=1).flatten()
        avg_width = np.mean(widths)
        bands = bands[widths >= 0.7 * avg_width]

        return bands


    def detect_answers(self, gray):
        h = gray.shape[0]
        gray = gray[int(0.05*h):, :] #to "cut" the top edge a bit off so it's not a closed contour and cv2.RETR_EXTERNAL doesn't detect only that

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow('gray', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
        binary = np.zeros(gray.shape, dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.4:
                cv2.drawContours(binary, [cnt], -1, 255, 2)



        cv2.imshow('bin', binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 1. Find the the y peaks, that enclose the circles (basically the islands in the projection
        col_proj = np.sum(binary, axis=0)


        from scipy.ndimage import uniform_filter1d
        from scipy.signal import find_peaks
        import matplotlib.pyplot as plt


        col_proj_smooth = uniform_filter1d(col_proj.astype(float), size=7)

        col_peaks, _ = find_peaks(
            col_proj_smooth,
            prominence = np.max(col_proj_smooth) * 0.05,
            distance=3
        )

        v_bands = self.detect_circle_bands(col_peaks, col_proj_smooth)

        band_centers = [(b[0]+b[1])//2 for b in v_bands]
        gaps = np.diff(band_centers)
        avg_gap = np.mean(gaps)

        split_idx = None
        for i, gap in enumerate(gaps):
            if gap > avg_gap * 2:
                split_idx = i + 1
                break

        question_cols = [v_bands[:split_idx], v_bands[split_idx:]] if split_idx else [v_bands]

        answers = {}
        current_question = 1

        #DEBUG
        total_circles = 0

        for question_col in question_cols:
            band_data = []

            for band in question_col:
                padding = 5
                x1 = max(band[0] - padding, 0)
                x2 = min(band[1] + padding, binary.shape[1])
                region = binary[:, x1:x2]

                row_proj = np.sum(region, axis=1)

                row_proj_smooth = uniform_filter1d(row_proj.astype(float), size=11)

                row_peaks,  _= find_peaks(
                    row_proj_smooth,
                    prominence = np.max(row_proj_smooth) * 0.1,
                    distance = 10
                )

                #plt.figure(figsize=(15, 4))
                #plt.plot(row_proj_smooth)
                #plt.scatter(row_peaks, row_proj_smooth[row_peaks], color='green', zorder=5, label='detected peaks')
                #plt.axhline(np.max(row_proj_smooth) * 0.28, color='red', linestyle='--', label='height threshold')
                #plt.xlabel('column index')
                #plt.legend()
                #plt.show()

                h_bands = self.detect_circle_bands(row_peaks, row_proj_smooth)
                total_circles += len(h_bands)

                band_data.append((band, h_bands))
        
            num_rows = max(len(r[1]) for r in band_data)

            for row in range(num_rows):
                fill_ratios = []

                coords = []
                for (x1, x2), h_bands in band_data:
                    coords.append((x1, x2, h_bands[row][0], h_bands[row][1]))

                avg_width = np.mean([x2 - x1 for x1, x2, _, _ in coords]) + 5 #padding
                avg_height = np.mean([y2 - y1 for _, _, y1, y2 in coords]) + 5 #padding

                for x1, x2, y1, y2 in coords:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    roi = gray[int(cy - avg_height / 2):int(cy + avg_height / 2), int(cx - avg_width / 2):int(cx + avg_width / 2)]
                    fill_ratios.append(1.0 - np.mean(roi) / 255.0)

                best = int(np.argmax(fill_ratios))
                if fill_ratios[best] < 0.3:
                    answers[str(current_question)] = ""
                else:
                    answers[str(current_question)] = "ABCDEF"[best]
                current_question += 1


        #plt.figure(figsize=(15, 4))
        #plt.plot(col_proj_smooth)
        #plt.scatter(col_peaks, col_proj_smooth[col_peaks], color='green', zorder=5, label='detected peaks')
        #plt.axhline(np.max(col_proj_smooth) * 0.28, color='red', linestyle='--', label='height threshold')
        #plt.xlabel('column index')
        #plt.legend()
        #plt.show()


        return answers

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
