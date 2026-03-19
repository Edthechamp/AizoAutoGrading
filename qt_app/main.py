import sys
from PyQt6.QtWidgets import QApplication
from window import MainWindow
from camera import CameraThread

app = QApplication(sys.argv)

camera_thread = CameraThread()
camera_thread.start()

window = MainWindow(camera_thread)
window.show()
sys.exit(app.exec())