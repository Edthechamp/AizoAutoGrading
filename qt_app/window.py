from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QLineEdit,
    QVBoxLayout, QWidget, QStackedWidget, QHBoxLayout,
    QInputDialog, QMessageBox
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import cv2
import numpy as np
from imutils.perspective import four_point_transform, order_points
import json

from extract import Extractor
import utils
import checkAns


class MainWindow(QMainWindow):
    def __init__(self, camera_thread):
        super().__init__()
        self.setWindowTitle("Labotajs")
        self.setFixedSize(QSize(1600, 900))

        self.event_name = ""

        self.is_drawing_corners = False
        self.is_camera_visible = False
        self.document_corners = []
        self.camera_rotation = 0

        self.topic_boxes = []
        self.current_topic_box = None
        self.current_topic_name = None

        self.code_box = None
        self.pending_code_box = None

        self.draw_mode = None
        self.warped_frame = None
        self.warped_pixmap = None

        self.drag_start = None
        self.drag_end = None

        self.offset_x = 0.0
        self.offset_y = 0.0
        self.scale_x = 1.0
        self.scale_y = 1.0

        self.extractor = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._build_event_name_submit_page())
        self.stack.addWidget(self._build_document_corner_draw_page())
        self.stack.addWidget(self._build_topic_draw_page())
        self.stack.addWidget(self._build_correct_ans_scan_page())
        self.stack.addWidget(self._build_test_scan_page())
        self.stack.addWidget(self._build_full_scan_page())

        self.thread = camera_thread
        self.thread.frame_ready.connect(self._update_frame)

    def _build_event_name_submit_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        label = QLabel("Ievadi notikuma nosaukumu (piemeram AIZO2026):")
        font = label.font()
        font.setPointSize(30)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        self.event_name = QLineEdit()
        self.event_name.setPlaceholderText("Notikuma nosaukums")
        layout.addWidget(self.event_name)

        btn = QPushButton("Iesniegt")
        btn.clicked.connect(self._on_event_name_submit)
        layout.addWidget(btn)

        page.setLayout(layout)
        return page

    def _build_document_corner_draw_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        btn_container = QWidget()
        btn_container.setStyleSheet("background-color: #2b2b2b;")
        btn_row = QHBoxLayout(btn_container)

        self.draw_btn = QPushButton("Atzimet dokumenta robezas")
        self.draw_btn.clicked.connect(self._document_draw_corners)

        self.rotate_btn = QPushButton("Pagriezt 90°")
        self.rotate_btn.clicked.connect(self._rotate_camera)

        self.reset_btn = QPushButton("Izdzest atzimetos punktus")
        self.reset_btn.hide()
        self.reset_btn.clicked.connect(self._document_reset_corners)

        self.confirm_btn = QPushButton("Turpinat")
        self.confirm_btn.hide()
        self.confirm_btn.clicked.connect(self._document_on_corner_submit)

        btn_row.addWidget(self.draw_btn)
        btn_row.addWidget(self.rotate_btn)
        btn_row.addWidget(self.reset_btn)
        btn_row.addWidget(self.confirm_btn)

        self.camera_label = CameraLabel("Gaida kameru...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.clicked.connect(self._document_on_camera_click)

        layout.addWidget(btn_container, stretch=15)
        layout.addWidget(self.camera_label, stretch=85)

        page.setLayout(layout)
        return page

    def _build_topic_draw_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        btn_container = QWidget()
        btn_container.setStyleSheet("background-color: #2b2b2b;")
        btn_row = QHBoxLayout(btn_container)

        self.add_topic_btn = QPushButton("Atzimet jaunu temata kastiti")
        self.add_topic_btn.clicked.connect(self._start_adding_topic)

        self.cancel_topic_btn = QPushButton("Atcelt temata pievienosanu")
        self.cancel_topic_btn.hide()
        self.cancel_topic_btn.clicked.connect(self._cancel_adding_topic)

        self.save_topic_btn = QPushButton("Saglabat temata kastiti")
        self.save_topic_btn.hide()
        self.save_topic_btn.clicked.connect(self._save_topic)

        self.continue_to_code_btn = QPushButton("Turpinat")
        self.continue_to_code_btn.hide()
        self.continue_to_code_btn.clicked.connect(self._continue_to_code)

        btn_row.addWidget(self.add_topic_btn)
        btn_row.addWidget(self.cancel_topic_btn)
        btn_row.addWidget(self.save_topic_btn)
        btn_row.addWidget(self.continue_to_code_btn)

        self.add_code_btn = QPushButton("Atzimet koda laukumu")
        self.add_code_btn.hide()
        self.add_code_btn.clicked.connect(self._start_adding_code)

        self.cancel_code_btn = QPushButton("Atcelt")
        self.cancel_code_btn.hide()
        self.cancel_code_btn.clicked.connect(self._cancel_adding_code)

        self.save_code_btn = QPushButton("Saglabat")
        self.save_code_btn.hide()
        self.save_code_btn.clicked.connect(self._save_code)

        self.continue_to_scan_btn = QPushButton("Turpinat")
        self.continue_to_scan_btn.hide()
        self.continue_to_scan_btn.clicked.connect(self._continue_to_scan)

        btn_row.addWidget(self.add_code_btn)
        btn_row.addWidget(self.cancel_code_btn)
        btn_row.addWidget(self.save_code_btn)
        btn_row.addWidget(self.continue_to_scan_btn)

        self.document_label = CameraLabel()
        self.document_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.document_label.pressed.connect(self._on_document_press)
        self.document_label.dragged.connect(self._on_document_drag)
        self.document_label.released.connect(self._on_document_release)

        layout.addWidget(btn_container, stretch=15)
        layout.addWidget(self.document_label, stretch=85)

        page.setLayout(layout)
        return page

    def _build_correct_ans_scan_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        btn_container = QWidget()
        btn_container.setStyleSheet("background-color: #2b2b2b;")
        btn_row = QHBoxLayout(btn_container)

        self.scan_correct_ans = QPushButton("Skenet pareizas atbildes")
        self.scan_correct_ans.clicked.connect(self._scan_correct_ans)

        self.rescan_btn = QPushButton("Meginat velreiz")
        self.rescan_btn.hide()
        self.rescan_btn.clicked.connect(self._scan_correct_ans)

        self.continue_to_test_btn = QPushButton("Turpinat")
        self.continue_to_test_btn.hide()
        self.continue_to_test_btn.clicked.connect(self._continue_to_test)

        btn_row.addWidget(self.scan_correct_ans)
        btn_row.addWidget(self.rescan_btn)
        btn_row.addWidget(self.continue_to_test_btn)

        self.scan_camera_label = CameraLabel("Gaida kameru...")
        self.scan_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(btn_container, stretch=15)
        layout.addWidget(self.scan_camera_label, stretch=85)

        page.setLayout(layout)
        return page

    def _build_test_scan_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        btn_container = QWidget()
        btn_container.setStyleSheet("background-color: #2b2b2b;")
        btn_row = QHBoxLayout(btn_container)

        self.test_scan_btn = QPushButton("Veikt testa skenesanu")
        self.test_scan_btn.clicked.connect(self._test_scan)

        self.continue_to_full_scan_btn = QPushButton("Turpinat")
        self.continue_to_full_scan_btn.clicked.connect(self._continue_to_full_scan)

        btn_row.addWidget(self.test_scan_btn)
        btn_row.addWidget(self.continue_to_full_scan_btn)

        self.test_scan_camera_label = CameraLabel("Gaida kameru...")
        self.test_scan_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(btn_container, stretch=15)
        layout.addWidget(self.test_scan_camera_label, stretch=85)

        page.setLayout(layout)
        return page

    def _build_full_scan_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        btn_container = QWidget()
        btn_container.setStyleSheet("background-color: #2b2b2b;")
        btn_row = QHBoxLayout(btn_container)

        self.full_scan_btn = QPushButton("Veikt visu darbu skenesanu")
        self.full_scan_btn.clicked.connect(self._full_scan)

        self.close_btn = QPushButton("Aizvērt")
        self.close_btn.hide()
        self.close_btn.clicked.connect(self.close)

        btn_row.addWidget(self.full_scan_btn)

        layout.addWidget(btn_container)

        page.setLayout(layout)
        return page

    def _on_event_name_submit(self):
        event_name = self.event_name.text().strip()
        if not event_name:
            return
        self.event_name = event_name
        self.stack.setCurrentIndex(1)
        self.is_camera_visible = True

    def _update_frame(self, frame):
        if not self.is_camera_visible:
            return

        if self.camera_rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.camera_rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.camera_rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.current_frame = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )

        self.offset_x = (self.camera_label.width() - pixmap.width()) / 2
        self.offset_y = (self.camera_label.height() - pixmap.height()) / 2
        self.scale_x = frame.shape[1] / pixmap.width()
        self.scale_y = frame.shape[0] / pixmap.height()

        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor(0, 255, 0), 3))

        if self.document_corners:
            pts = [(int(x / self.scale_x), int(y / self.scale_y))
                   for x, y in self.document_corners]
            for px, py in pts:
                painter.drawEllipse(px - 6, py - 6, 12, 12)
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
            if len(pts) == 4:
                painter.drawLine(pts[3][0], pts[3][1], pts[0][0], pts[0][1])

        painter.end()

        idx = self.stack.currentIndex()
        if idx == 1:
            self.camera_label.setPixmap(pixmap)
        elif idx == 3:
            self.scan_camera_label.setPixmap(pixmap)
        elif idx == 4:
            self.test_scan_camera_label.setPixmap(pixmap)

    def _rotate_camera(self):
        self.camera_rotation = (self.camera_rotation + 90) % 360
        self.document_corners = []
        self.confirm_btn.setEnabled(False)

    def _document_draw_corners(self):
        self.draw_btn.hide()
        self.reset_btn.show()
        self.confirm_btn.show()
        self.confirm_btn.setEnabled(False)
        self.is_drawing_corners = True

    def _document_on_camera_click(self, x, y):
        if not self.is_drawing_corners or len(self.document_corners) >= 4:
            return
        self.document_corners.append(
            [int((x - self.offset_x) * self.scale_x),
             int((y - self.offset_y) * self.scale_y)]
        )
        if len(self.document_corners) == 4:
            self.confirm_btn.setEnabled(True)

    def _document_reset_corners(self):
        self.document_corners = []
        self.confirm_btn.setEnabled(False)

    def _document_on_corner_submit(self):
        self.warped_frame = self._warp_document(self.current_frame, self.document_corners)
        self.document_corners = (
            order_points(np.array(self.document_corners, dtype="float32"))
            .astype(int).tolist()
        )

        rgb = cv2.cvtColor(self.warped_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.warped_pixmap = QPixmap.fromImage(qimg)

        self.stack.setCurrentIndex(2)
        self.is_camera_visible = False
        self._update_document_frame()

    def _warp_document(self, frame, corners):
        raw_pts = np.array([(x, y) for x, y in corners], dtype="float32")
        return four_point_transform(frame, raw_pts)

    def _start_adding_topic(self):
        self.draw_mode = "topic"
        self.current_topic_box = None
        self.drag_start = None
        self.drag_end = None
        self.add_topic_btn.hide()
        self.cancel_topic_btn.show()
        self.save_topic_btn.show()
        self.save_topic_btn.setEnabled(False)
        self.continue_to_code_btn.hide()

        name, ok = QInputDialog.getText(self, "Temas nosaukums", "Ievadi temas nosaukumu:")
        if not ok or not name.strip():
            self._cancel_adding_topic()
            return
        self.current_topic_name = name

    def _cancel_adding_topic(self):
        self.current_topic_box = None
        self.current_topic_name = None
        self.drag_start = None
        self.drag_end = None
        self.draw_mode = None
        self.add_topic_btn.show()
        self.cancel_topic_btn.hide()
        self.save_topic_btn.hide()
        self._update_document_frame()

        if self.topic_boxes:
            self.continue_to_code_btn.show()

    def _save_topic(self):
        self.topic_boxes.append({
            "label": self.current_topic_name.strip(),
            "box": self.current_topic_box,
        })
        self._cancel_adding_topic()
        self.continue_to_code_btn.show()

    def _continue_to_code(self):
        self.add_topic_btn.hide()
        self.cancel_topic_btn.hide()
        self.save_topic_btn.hide()
        self.add_code_btn.show()
        self.draw_mode = None
        self.continue_to_code_btn.hide()

    def _start_adding_code(self):
        self.draw_mode = "code"
        self.pending_code_box = None
        self.drag_start = None
        self.drag_end = None
        self.add_code_btn.hide()
        self.cancel_code_btn.show()
        self.save_code_btn.show()
        self.save_code_btn.setEnabled(False)
        self.continue_to_scan_btn.hide()

    def _cancel_adding_code(self):
        self.draw_mode = None
        self.pending_code_box = None
        self.drag_start = None
        self.drag_end = None
        self.add_code_btn.show()
        self.cancel_code_btn.hide()
        self.save_code_btn.hide()
        self._update_document_frame()

    def _save_code(self):
        self.draw_mode = None
        self.code_box = self.pending_code_box
        self.pending_code_box = None
        self.drag_start = None
        self.drag_end = None
        self.add_code_btn.show()
        self.cancel_code_btn.hide()
        self.save_code_btn.hide()
        self._update_document_frame()
        self.continue_to_scan_btn.show()

    def _continue_to_scan(self):
        self.extractor = Extractor(
            self.thread, self.document_corners, self.topic_boxes, self.code_box, self.camera_rotation
        )
        self.is_camera_visible = True
        self.stack.setCurrentIndex(3)

    def _to_image_coords(self, widget_x, widget_y):
        return (
            int((widget_x - self.offset_x) * self.scale_x),
            int((widget_y - self.offset_y) * self.scale_y),
        )

    def _on_document_press(self, x, y):
        if self.draw_mode not in ("topic", "code"):
            return

        if self.draw_mode == "topic":
            self.current_topic_box = None
            self.save_topic_btn.setEnabled(False)
        else:
            self.pending_code_box = None
            self.save_code_btn.setEnabled(False)

        self.drag_start = self._to_image_coords(x, y)
        self.drag_end = self.drag_start
        self._update_document_frame()

    def _on_document_drag(self, x, y):
        if self.drag_start is None:
            return
        self.drag_end = self._to_image_coords(x, y)
        self._update_document_frame()

    def _on_document_release(self, x, y):
        if self.drag_start is None:
            return

        self.drag_end = self._to_image_coords(x, y)

        x1, y1 = self.drag_start
        x2, y2 = self.drag_end

        box = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]

        if self.draw_mode == "topic":
            self.current_topic_box = box
            self.save_topic_btn.setEnabled(True)
        elif self.draw_mode == "code":
            self.pending_code_box = box
            self.save_code_btn.setEnabled(True)

        self.drag_start = None
        self.drag_end = None
        self._update_document_frame()

    def _update_document_frame(self):
        if self.warped_pixmap is None:
            return

        pixmap = self.warped_pixmap.scaled(
            self.document_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.offset_x = (self.document_label.width() - pixmap.width()) / 2
        self.offset_y = (self.document_label.height() - pixmap.height()) / 2
        self.scale_x = self.warped_frame.shape[1] / pixmap.width()
        self.scale_y = self.warped_frame.shape[0] / pixmap.height()

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        def draw_rect(box, color, label_text=""):
            if box is None:
                return
            bx, by, bw, bh = box
            dx = int(bx / self.scale_x)
            dy = int(by / self.scale_y)
            dw = int(bw / self.scale_x)
            dh = int(bh / self.scale_y)

            painter.setPen(QPen(color, 2))
            painter.drawRect(dx, dy, dw, dh)

            if label_text:
                font = QFont()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QPen(color, 1))
                painter.drawText(dx + 4, dy + 16, label_text)

        for tb in self.topic_boxes:
            draw_rect(tb["box"], QColor(255, 140, 0), tb["label"])

        draw_rect(self.current_topic_box, QColor(0, 255, 0))

        draw_rect(self.code_box, QColor(0, 0, 255))
        draw_rect(self.pending_code_box, QColor(0, 0, 255))

        if self.drag_start and self.drag_end:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            preview = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
            preview_color = QColor(0, 255, 0) if self.draw_mode == "topic" else QColor(0, 0, 255)
            draw_rect(preview, preview_color)

        painter.end()
        self.document_label.setPixmap(pixmap)

    def _scan_correct_ans(self):
        answers = self.extractor.scan_answers()

        QMessageBox.information(self, "correct answers", "Atbildes:\n" + str(answers))

        self.scan_correct_ans.hide()
        self.rescan_btn.show()
        self.continue_to_test_btn.show()

        with open(utils.getFilePath("answers.json"), "w") as f:
            json.dump(answers, f, indent=4)
            print("answers saved to file")

    def _continue_to_test(self):
        self.stack.setCurrentIndex(4)

    def _test_scan(self):
        if utils.dispensePage():
            answers = self.extractor.scan_answers()
            QMessageBox.information(self, "scanned answers", "Atbildes:\n" + str(answers))
        else:
            QMessageBox.critical(
                self, "ERROR",
                "Failed to dispense test page, enter paper please or check printer"
            )

    def _continue_to_full_scan(self):
        self.stack.setCurrentIndex(5)

    def _full_scan(self):
        self.full_scan_btn.hide()

        while True:
            if utils.dispensePage():
                try:
                    answers = self.extractor.scan_answers()
                except Exception:
                    continue

                test = {
                    "studentID": answers.get("code", ""),
                    "answers": {k: v for k, v in answers.items() if k != "code"},
                }
                print("temp structure, passing to Grade", test)
                grades = checkAns.GradeTest(test)
                print("graded:", grades)
                utils.saveAnswers(grades)
            else:
                QMessageBox.information(
                    self, "scanned answers",
                    "Visas atbildes ir ieskenetas un saglabatas ..."
                )
                break

        self.close_btn.show()


class CameraLabel(QLabel):
    clicked  = pyqtSignal(int, int)
    pressed  = pyqtSignal(int, int)
    dragged  = pyqtSignal(int, int)
    released = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            x, y = int(event.position().x()), int(event.position().y())
            self.clicked.emit(x, y)
            self.pressed.emit(x, y)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.dragged.emit(
                int(event.position().x()),
                int(event.position().y()),
            )

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.released.emit(
                int(event.position().x()),
                int(event.position().y()),
            )
