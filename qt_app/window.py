from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QVBoxLayout, QWidget, QStackedWidget, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
import cv2


class MainWindow(QMainWindow):
    def __init__(self, camera_thread):
        super().__init__()
        self.setWindowTitle("Labotajs")
        self.setFixedSize(QSize(1600, 900))

        #shared state
        self.event_name = ""
        self.corners = []

        #stack
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._build_event_name_submit_page())
        self.stack.addWidget(self._build_document_corner_draw_page())

        #camera
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
        btn_container.setStyleSheet("background-color: #2b2b2b;")     #Nomainit lai smuki velak
        btn_row = QHBoxLayout(btn_container)

        self.draw_btn = QPushButton("Atzimet dokumenta robezas")
        self.draw_btn.clicked.connect(self._draw_corners)

        self.reset_btn = QPushButton("Izdzest atzimetos punktus")
        #self.reset_btn.setEnabled(False)
        self.reset_btn.hide()
        self.reset_btn.clicked.connect(self._reset_corners)

        self.confirm_btn = QPushButton("Turpinat")
        #self.confirm_btn.setEnabled(False)
        self.confirm_btn.hide()
        self.confirm_btn.clicked.connect(self._on_corner_submit)

        btn_row.addWidget(self.draw_btn)
        btn_row.addWidget(self.reset_btn)
        btn_row.addWidget(self.confirm_btn)


        self.camera_label = QLabel("Gaida kameru...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(btn_container, stretch=15)
        layout.addWidget(self.camera_label, stretch=85)

        page.setLayout(layout)


        return page

    #states
    def _update_frame(self, frame):
        self.current_frame = frame

        # convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        # convert to QImage then QPixmap
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # scale to fit the label while keeping aspect ratio
        pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )

        self.camera_label.setPixmap(pixmap)


    #widget0 - event_name
    def _on_event_name_submit(self):
        event_name = self.event_name.text().strip()
        if not event_name:
            return
        self.event_name = event_name
        self.stack.setCurrentIndex(1)
        print(f"izdevaas {self.event_name}")
    
    #widget1 - document_corners
    def _draw_corners(self):
        self.thread.toggle_pause()
        self.draw_btn.setEnabled(False)
        pass

    def _reset_corners(self):
        self.corners = []
        self.confirm_btn.setEnabled(False)

    def _on_corner_submit(self):
        pass
    
