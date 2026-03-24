from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QVBoxLayout, QWidget, QStackedWidget, QHBoxLayout, QInputDialog, QMessageBox
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import cv2
import numpy as np
from imutils.perspective import four_point_transform, order_points
from extract import Extractor


class MainWindow(QMainWindow):
    def __init__(self, camera_thread):
        super().__init__()
        self.setWindowTitle("Labotajs")
        self.setFixedSize(QSize(1600, 900))

        #page 0 state
        self.event_name = ""

        #page 1 state
        self.is_drawing_corners = False
        self.is_camera_visible = False
        self.document_corners = []

        #page 2 state
        self.topic_boxes = [] #[{"label": str, "pts": [[x,y]x4]}, ...]
        self.current_topic_pts = []
        self.current_topic_name = None
        self.code_corners = []
        self.draw_mode = None
        self.warped_frame = None
        self.warped_pixmap = None

        #page 3 state
        self.extractor = None

        #stack
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._build_event_name_submit_page())
        self.stack.addWidget(self._build_document_corner_draw_page())
        self.stack.addWidget(self._build_topic_draw_page())
        self.stack.addWidget(self._build_correct_ans_scan_page())
        self.stack.addWidget(self._build_test_scan_page())
        self.stack.addWidget(self._build_full_scan_page())

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
        self.draw_btn.clicked.connect(self._document_draw_corners)

        self.reset_btn = QPushButton("Izdzest atzimetos punktus")
        self.reset_btn.hide()
        self.reset_btn.clicked.connect(self._document_reset_corners)

        self.confirm_btn = QPushButton("Turpinat")
        self.confirm_btn.hide()
        self.confirm_btn.clicked.connect(self._document_on_corner_submit)

        btn_row.addWidget(self.draw_btn)
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
        btn_container.setStyleSheet("background-color: #2b2b2b;")     #Nomainit lai smuki velak
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
        self.document_label.clicked.connect(self._on_document_click)

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
        btn_container.setStyleSheet("background-color: #2b2b2b;")     #Nomainit lai smuki velak
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
        self.scan_camera_label.clicked.connect(self._document_on_camera_click)

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
        btn_container.setStyleSheet("background-color: #2b2b2b;")     #Nomainit lai smuki velak
        btn_row = QHBoxLayout(btn_container)

        self.test_scan_btn = QPushButton("Veikt testa skenesanu")
        self.test_scan_btn.clicked.connect(self._test_scan)

        self.continue_to_full_scan_btn = QPushButton("Turpinat")
        self.continue_to_full_scan_btn.clicked.connect(self._continue_to_full_scan)

        btn_row.addWidget(self.test_scan_btn)
        btn_row.addWidget(self.continue_to_full_scan_btn)

        self.test_scan_camera_label = CameraLabel("Gaida kameru...")
        self.test_scan_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.test_scan_camera_label.clicked.connect(self._document_on_camera_click)

        layout.addWidget(btn_container, stretch=15)
        layout.addWidget(self.scan_camera_label, stretch=85)


        page.setLayout(layout)
        return page

    def _build_full_scan_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        btn_container = QWidget()
        btn_container.setStyleSheet("background-color: #2b2b2b;")     #Nomainit lai smuki velak
        btn_row = QHBoxLayout(btn_container)

        self.full_scan_btn = QPushButton("Veikt visu darbu skenesanu")
        self.test_scan_btn.clicked.connect(self._full_scan)

        self.close_btn = QPushButton("Aizvērt")
        self.close_btn.hide()
        self.close_btn.clicked.connect(self.close)

        btn_row.addWidget(self.full_scan_btn)

        layout.addWidget(btn_container)


        page.setLayout(layout)
        return page



    #--------
    # HANDLERS
    # -------

    #PAGE 0
    def _on_event_name_submit(self):
        event_name = self.event_name.text().strip()
        if not event_name:
            return
        self.event_name = event_name
        self.stack.setCurrentIndex(1)
        self.is_camera_visible = True


    #PAGE 1
    def _update_frame(self, frame):
        if self.is_camera_visible:
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

            self.offset_x = (self.camera_label.width() - pixmap.width()) / 2
            self.offset_y = (self.camera_label.height() - pixmap.height()) / 2
            self.scale_x = self.thread.width / pixmap.width()
            self.scale_y = self.thread.height / pixmap.height()

            # draw corners
            if self.document_corners:
                painter = QPainter(pixmap)
                pen = QPen(QColor(0, 255, 0), 3)
                painter.setPen(pen)

                
                pts = [(int(x / self.scale_x), int(y / self.scale_y)) for x, y in self.document_corners]

                for px, py in pts:
                    painter.drawEllipse(px - 6, py - 6, 12, 12)
                for i in range(len(pts) - 1):
                    painter.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                if len(pts) == 4:
                    painter.drawLine(pts[3][0], pts[3][1], pts[0][0], pts[0][1])

                painter.end()
            if self.stack.currentIndex() == 1:
                self.camera_label.setPixmap(pixmap)
            elif self.stack.currentIndex() == 3:
                self.scan_camera_label.setPixmap(pixmap)
            elif self.stack.currentIndex() == 4:
                self.test_scan_camera_label.setPixmap(pixmap)
    
    def _document_draw_corners(self):
        self.draw_btn.hide()
        self.reset_btn.show()
        self.confirm_btn.show()
        self.confirm_btn.setEnabled(False) #also shows the button
        self.is_drawing_corners = True
    
    def _document_on_camera_click(self, x, y):
        if not self.is_drawing_corners or len(self.document_corners) >= 4:
            return
        self.document_corners.append([int((x - self.offset_x) * self.scale_x), int((y - self.offset_y) * self.scale_y)])
        if len(self.document_corners) == 4:
            self.confirm_btn.setEnabled(True)

    def _document_reset_corners(self):
        self.document_corners = []
        self.confirm_btn.setEnabled(False)

    def _document_on_corner_submit(self):
        #warp current frame using the 4 drawn corners
        self.warped_frame = self._warp_document(self.current_frame, self.document_corners)
        self.document_corners = order_points(np.array(self.document_corners, dtype="float32")).astype(int)

        #convert to QPixMap and cache it
        rgb = cv2.cvtColor(self.warped_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.warped_pixmap = QPixmap.fromImage(qimg)

        self.stack.setCurrentIndex(2)
        self.is_camera_visible = False
        self._update_document_frame()
    
    def _warp_document(self, frame, corners):
        raw_pts = np.array([(x, y) for x, y in corners], dtype="float32")
        warped = four_point_transform(frame, raw_pts)

        return warped

   #PAGE 2 
    def _start_adding_topic(self):
        self.draw_mode = "topic"
        self.current_topic_pts = []
        self.add_topic_btn.hide()
        self.cancel_topic_btn.show()
        self.save_topic_btn.show()
        self.save_topic_btn.setEnabled(False)

        name, ok = QInputDialog.getText(self, "Temas nosaukums", "Ievadi temas nosaukumu:")
        if not ok or not name.strip():
            return

        self.current_topic_name = name

        self.continue_to_code_btn.hide()

    def _cancel_adding_topic(self):
        self.current_topic_pts = []
        self.current_topic_name = None
        self.draw_mode = None
        self.add_topic_btn.show()
        self.cancel_topic_btn.hide()
        self.save_topic_btn.hide()
        self._update_document_frame()

        if self.topic_boxes:
            self.continue_to_code_btn.show()

    def _save_topic(self):
        self.topic_boxes.append({"label": self.current_topic_name.strip(), "pts": list(order_points(np.array(self.current_topic_pts, dtype="float32")).astype(int))})
        #cool trick to save rewriting code
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
        self.code_corners = []
        self.add_code_btn.hide()
        self.cancel_code_btn.show()
        self.save_code_btn.show()
        self.save_code_btn.setEnabled(False)

        self.continue_to_scan_btn.hide()
    
    def _cancel_adding_code(self):
        self.draw_mode = None
        self.code_corners = []
        self.add_code_btn.show()
        self.cancel_code_btn.hide()
        self.save_code_btn.hide()
        self._update_document_frame()
    
    def _save_code(self):
        self.draw_mode = None
        self.code_corners = list(order_points(np.array(self.code_corners, dtype="float32")).astype(int))
        self.add_code_btn.show()
        self.cancel_code_btn.hide()
        self.save_code_btn.hide()
        self._update_document_frame()

        self.continue_to_scan_btn.show()
    
    def _continue_to_scan(self):
        self.extractor = Extractor(self.thread, self.document_corners, self.topic_boxes, self.code_corners)
        self.is_camera_visible = True
        self.stack.setCurrentIndex(3)

    def _on_document_click(self, x, y):
        if self.draw_mode == "topic":
            if len(self.current_topic_pts) >= 4:
                return
            self.current_topic_pts.append([int((x - self.offset_x) * self.scale_x), int((y - self.offset_y) * self.scale_y)])
            if len(self.current_topic_pts) == 4:
                self.save_topic_btn.setEnabled(True)
            self._update_document_frame()
        elif self.draw_mode == "code":
            if len(self.code_corners) >= 4:
                return
            self.code_corners.append([int((x - self.offset_x) * self.scale_x), int((y - self.offset_y) * self.scale_y)])
            if len(self.code_corners) == 4:
                self.save_code_btn.setEnabled(True)
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

        def draw_quad(pts, color, label_text=""):
            pen = QPen(color, 2)
            painter.setPen(pen)
            scaled = [(int((x / self.scale_x)), int((y / self.scale_y))) for x, y in pts]
            for px, py in scaled:
                painter.drawEllipse(px - 5, py - 5, 10, 10)
            for i in range(len(scaled) - 1):
                painter.drawLine(scaled[i][0], scaled[i][1], scaled[i+1][0], scaled[i+1][1])
            if len(scaled) == 4:
                painter.drawLine(scaled[3][0], scaled[3][1], scaled[0][0], scaled[0][1])
                if label_text:
                    font = QFont()
                    font.setPointSize(10)
                    font.setBold(True)
                    painter.setFont(font)
                    painter.setPen(QPen(color, 1))
                    painter.drawText(scaled[0][0] + 4, scaled[0][1] + 16, label_text)
        
        for box in self.topic_boxes:
            draw_quad(box["pts"], QColor(255, 140, 0)) #orange
        
        if self.current_topic_pts:
            draw_quad(self.current_topic_pts, QColor(0, 255, 0))
        
        if self.code_corners:
            draw_quad(self.code_corners, QColor(0, 0, 255))
        
        painter.end()
        self.document_label.setPixmap(pixmap)

    #PAGE 3
    def _scan_correct_ans(self):
        #DOESNT WORK CURRENTLY
        answers = self.extractor.scan_answers()

        QMessageBox.information(self, "correct answers", "Atbildes:\n"+str(answers))

        self.scan_correct_ans.hide()
        self.rescan_btn.show()
        self.continue_to_test_btn.show()

        #Add something here
        #MAZERS COMPLETE SAVING ANSWERS

    #--------
    # PAGE 4
    #--------

    def _continue_to_test(self):
        self.stack.setCurrentIndex(4)

    def _test_scan(self):
        #IZPRINTE 1 LAPU
        #...

        answers = self.extractor.scan_answers()

        QMessageBox.information(self, "scanned answers", "Atbildes:\n"+str(answers))


    def _continue_to_full_scan(self):
        self.stack.setCurrentIndex(5)

    #--------
    # PAGE 5
    #--------

    def _full_scan(self):
        self.full_scan_btn.hide()
        # i IS FOR TESTING
        i = 0
        while True:
            #IZPRITNE 1 LAPU
            #...
            i+=1
            if i > 500:
                break

            answers = self.extractor.scan_answers()

            #IZDOMA KO DARIT AR ATBILDEM, SAGLABAT VAI UZ VIETAS PARBAUDA

        #AFTER SCAN IS DONE
        #SAVE ANSWERS
        QMessageBox.information(self, "scanned answers", "Visas atbildes ir ieskenetas un saglabatas ...")

        self.close_btn.show()





class CameraLabel(QLabel):
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(int(event.position().x()), int(event.position().y()))
