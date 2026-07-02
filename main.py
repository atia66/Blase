import sys
from multiprocessing import Process, Queue

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont
from tables import drop_tables ,export_grads_to_csv
from dotenv import load_dotenv
from interface.styles import FONT_MAIN, C_BG_BASE, C_TEAL
from interface.welcome import WelcomeScreen
from interface.dashboard import DashboardScreen
import os
import psycopg2

load_dotenv()
DSN = os.getenv("DSN")
conn = psycopg2.connect(DSN)
conn.autocommit = True
def run_server_proc(log_queue: Queue):
    from Machine.app import run_server
    log_queue.put("SERVER: starting…")
    run_server()

def run_written_worker(log_queue: Queue):
    from written_worker import written_consumer
    log_queue.put("WRITTEN WORKER: starting…")
    written_consumer(DSN, log_queue)

def run_grading_worker(log_queue: Queue):
    from grading_worker import grading_consumer
    log_queue.put("GRADING WORKER: starting…")
    grading_consumer(DSN, log_queue)

def run_complete_worker(log_queue: Queue):
    from complete_worker import complete_consumer
    log_queue.put("COMPLETE WORKER: starting…")
    complete_consumer(DSN, log_queue)

def run_detection_worker(log_queue: Queue):
    from detection_worker import detection_consumer
    log_queue.put("DETECTION WORKER: starting…")
    detection_consumer(DSN, log_queue)

class LogMonitor(QThread):
    log_received     = Signal(str, str)   
    counter_received = Signal(str, int)   

    def __init__(self, log_queue: Queue, parent=None):
        super().__init__(parent)
        self._queue   = log_queue
        self._running = True

    def run(self):
        while self._running:
            while True:
                try:
                    msg = self._queue.get_nowait()

                    if isinstance(msg, tuple) and len(msg) == 3 and msg[0] == "__counter__":
                        _, name, value = msg
                        self.counter_received.emit(name, value)

                    elif isinstance(msg, tuple) and len(msg) == 2:
                        text, color = msg
                        self.log_received.emit(str(text), str(color))

                    else:
                        self.log_received.emit(str(msg), C_TEAL)

                except Exception:
                    break
            self.msleep(100)

    def stop(self):
        self._running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BLASE")
        self.setMinimumSize(960, 640)
        self.setStyleSheet(f"QMainWindow {{ background:{C_BG_BASE}; }}")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.welcome   = WelcomeScreen()
        self.dashboard = DashboardScreen()
        self.stack.addWidget(self.welcome)
        self.stack.addWidget(self.dashboard)


        self.welcome.start_btn.clicked.connect(self._new_session)
        self.welcome.continue_btn.clicked.connect(self._recover_session)   # ← changed
        self.dashboard.stop_btn.clicked.connect(self._stop_session)
        self.dashboard.export_btn.clicked.connect(self._export_csv)        # ← add this
        self._processes: list[Process]    = []
        self._log_queue: Queue | None     = None
        self._monitor:   LogMonitor | None = None

        self._center()

    def _new_session(self):
        self.dashboard.configure("new")
        self.stack.setCurrentWidget(self.dashboard)
        self._start_workers()
    def _recover_session(self):
        self._export_csv()
        
    def _export_csv(self):
        try:
            conn = psycopg2.connect(DSN)
            export_grads_to_csv(conn)
            conn.close()
        except Exception as e:
            print(f"Export failed: {e}")

    def _continue_session(self):
        self.dashboard.configure("resumed")
        self.stack.setCurrentWidget(self.dashboard)
        self._start_workers()

    def _stop_session(self):
        self._kill_workers()
        self.stack.setCurrentWidget(self.welcome)

    def _start_workers(self):
        self._kill_workers() 
        
        drop_tables()

        self._log_queue = Queue()

        targets = [
            run_server_proc,
            run_complete_worker,
            run_detection_worker,
            run_written_worker,
            run_grading_worker
        ]
        # for i in range(3) :
        #     targets.append(run_written_worker)
        #     targets.append(run_grading_worker)

        self._processes = [
            Process(target=fn, args=(self._log_queue,), daemon=True)
            for fn in targets
        ]
        for p in self._processes:
            p.start()

        self._monitor = LogMonitor(self._log_queue)
        self._monitor.log_received.connect(self.dashboard.add_log)
        self._monitor.counter_received.connect(self.dashboard.set_counter)
        self._monitor.start()

    def _kill_workers(self):
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor.wait(500)
            self._monitor = None

        for p in self._processes:
            if p.is_alive():
                p.terminate()
        for p in self._processes:
            p.join(timeout=3)
            if p.is_alive():
                p.kill()
        self._processes.clear()

        if self._log_queue is not None:
            self._log_queue.close()
            self._log_queue = None

    def closeEvent(self, event):
        self._kill_workers()
        super().closeEvent(event)

    def _center(self):
        geo = QApplication.primaryScreen().geometry()
        fg  = self.frameGeometry()
        fg.moveCenter(geo.center())
        self.move(fg.topLeft())

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont(FONT_MAIN, 10))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
