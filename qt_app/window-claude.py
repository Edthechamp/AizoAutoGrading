import cv2
import time
from PyQt6.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget,
    QVBoxLayout, QLabel, QLineEdit, QPushButton
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap


class CameraThread(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
            time.sleep(1 / 30)
        cap.release()

    def stop(self):
        self._running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document Scanner")
        self.resize(1000, 700)

        # shared state
        self.event_name = ""
        self.corners = []
        self.frozen_frame = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._build_name_page())    # 0
        self.stack.addWidget(self._build_camera_page())  # 1
        self.stack.addWidget(self._build_confirm_page()) # 2
        self.stack.setCurrentIndex(0)                    # <-- the fix

        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self._on_frame)
        self.camera_thread.start()

    def _build_name_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(200, 150, 200, 150)
        layout.setSpacing(12)

        layout.addWidget(QLabel("Event name:"))

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. event1")
        layout.addWidget(self.name_input)

        btn = QPushButton("Next →")
        btn.clicked.connect(self._on_name_confirmed)
        layout.addWidget(btn)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def _build_camera_page(self):
        page = QWidget()
        layout = QVBoxLayout()

        self.camera_label = QLabel("Waiting for camera...")
        self.camera_label.setMinimumHeight(400)
        layout.addWidget(self.camera_label)

        btn_row = QWidget()
        btn_layout = QVBoxLayout()

        self.confirm_btn = QPushButton("Confirm corners")
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self._on_corners_confirmed)

        reset_btn = QPushButton("Reset corners")
        reset_btn.clicked.connect(self._reset_corners)

        btn_layout.addWidget(self.confirm_btn)
        btn_layout.addWidget(reset_btn)
        btn_row.setLayout(btn_layout)
        layout.addWidget(btn_row)

        page.setLayout(layout)
        return page

    def _build_confirm_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(200, 150, 200, 150)
        layout.setSpacing(12)

        layout.addWidget(QLabel("Ready to scan:"))

        scan_btn = QPushButton("Scan")
        scan_btn.clicked.connect(self._on_scan)
        layout.addWidget(scan_btn)

        redo_btn = QPushButton("← Redo corners")
        redo_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        layout.addWidget(redo_btn)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def _on_frame(self, frame):
        self.current_frame = frame

        # convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        # convert to QImage then QPixmap
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # scale to fit the label while keeping aspect ratio
        pixmap = pixmap.scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        )

        self.camera_label.setPixmap(pixmap)

    def _on_name_confirmed(self):
        name = self.name_input.text().strip()
        if not name:
            return
        self.event_name = name
        self.stack.setCurrentIndex(1)

    def _on_corners_confirmed(self):
        self.frozen_frame = self.current_frame
        self.camera_thread.stop()
        self.stack.setCurrentIndex(2)

    def _reset_corners(self):
        self.corners = []
        self.confirm_btn.setEnabled(False)

    def _on_scan(self):
        from qt_app.camera import warp, enhance, save
        warped = warp(self.frozen_frame, self.corners)
        result = enhance(warped)
        save(result, self.event_name)
        self.stack.setCurrentIndex(1)

    def closeEvent(self, event):
        self.camera_thread.stop()
        self.camera_thread.wait()
        event.accept()