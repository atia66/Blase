from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame
)
from PySide6.QtCore import Qt, QTimer, QDateTime

from interface.styles import (
    font, h_line, solid_btn,
    C_BG_DEEP, C_BG_BASE, C_BG_TOPBAR,
    C_BORDER, C_TEXT_DIM, C_TEXT_PRI,
    C_BLUE, C_PURPLE, C_AMBER, C_TEAL,
    C_RED_BG, C_RED_FG, C_EXPBTN_BG, C_EXPBTN_FG,
)
from interface.widgets import CounterCard, LogItem, EmptyLogPlaceholder, TopBar, BottomBar


class DashboardScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{C_BG_BASE};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.topbar = TopBar(show_breadcrumb=True)
        root.addWidget(self.topbar)

        header_area = QWidget()
        header_area.setStyleSheet(f"background:{C_BG_BASE};")
        header_lay = QVBoxLayout(header_area)
        header_lay.setContentsMargins(28, 20, 28, 0)
        header_lay.setSpacing(14)

        sec1 = QLabel("PIPELINE COUNTERS")
        sec1.setFont(font(9, bold=True))
        sec1.setStyleSheet(
            f"color:{C_TEXT_DIM}; letter-spacing:3px; background:transparent;"
        )
        header_lay.addWidget(sec1)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(14)

        self.card_requests = CounterCard("Requests",      C_BLUE,   "Total received")
        self.card_students = CounterCard("Students",      C_PURPLE, "Enrolled this run")
        self.card_model    = CounterCard("Model Answers", C_AMBER,  "Loaded into pipeline")
        self.card_graded   = CounterCard("Graded",        C_TEAL,   "Awaiting completion")

        self._cards = {
            "requests": self.card_requests,
            "students": self.card_students,
            "model":    self.card_model,
            "graded":   self.card_graded,
        }
        for card in self._cards.values():
            cards_row.addWidget(card)

        header_lay.addLayout(cards_row)
        header_lay.addSpacing(6)
        header_lay.addWidget(h_line())
        root.addWidget(header_area)

        log_label_bar = QWidget()
        log_label_bar.setStyleSheet(f"background:{C_BG_BASE};")
        log_label_lay = QHBoxLayout(log_label_bar)
        log_label_lay.setContentsMargins(28, 14, 28, 8)

        log_sec = QLabel("ACTIVITY LOG")
        log_sec.setFont(font(9, bold=True))
        log_sec.setStyleSheet(
            f"color:{C_TEXT_DIM}; letter-spacing:3px; background:transparent;"
        )
        log_label_lay.addWidget(log_sec)
        log_label_lay.addStretch()

        self.log_count_lbl = QLabel("0 entries")
        self.log_count_lbl.setFont(font(9))
        self.log_count_lbl.setStyleSheet(f"color:{C_TEXT_DIM}; background:transparent;")
        log_label_lay.addWidget(self.log_count_lbl)
        root.addWidget(log_label_bar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setStyleSheet(f"""
            QScrollArea {{ background:{C_BG_BASE}; border:none; }}
            QScrollBar:vertical {{
                background:{C_BG_DEEP}; width:6px; border-radius:3px;
            }}
            QScrollBar::handle:vertical {{
                background:{C_BORDER}; border-radius:3px; min-height:30px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height:0px; }}
        """)

        self.log_container = QWidget()
        self.log_container.setStyleSheet(f"background:{C_BG_BASE};")
        self.log_layout = QVBoxLayout(self.log_container)
        self.log_layout.setContentsMargins(28, 4, 28, 20)
        self.log_layout.setSpacing(0)

        self._placeholder = EmptyLogPlaceholder()
        self.log_layout.addWidget(self._placeholder)
        self.log_layout.addStretch()

        self.scroll.setWidget(self.log_container)
        root.addWidget(self.scroll, 1)

        ctrl_bar = QWidget()
        ctrl_bar.setFixedHeight(60)
        ctrl_bar.setStyleSheet(
            f"background:{C_BG_TOPBAR}; border-top:1px solid {C_BORDER};"
        )
        ctrl_lay = QHBoxLayout(ctrl_bar)
        ctrl_lay.setContentsMargins(28, 0, 28, 0)
        ctrl_lay.setSpacing(12)

        self.status_lbl = QLabel("● Idle — start a session to begin")
        self.status_lbl.setFont(font(11))
        self.status_lbl.setStyleSheet(f"color:{C_TEXT_DIM}; background:transparent;")
        ctrl_lay.addWidget(self.status_lbl)
        ctrl_lay.addStretch()

        self.export_btn = solid_btn(
            "Export CSV", C_EXPBTN_BG, C_EXPBTN_FG, "#243D70",
            min_w=130, min_h=38, fsize=11,
        )
        ctrl_lay.addWidget(self.export_btn)

        self.stop_btn = solid_btn(
            "Stop session", C_RED_BG, C_RED_FG, "#991B1B",
            min_w=130, min_h=38, fsize=11,
        )
        ctrl_lay.addWidget(self.stop_btn)

        root.addWidget(ctrl_bar)
        root.addWidget(BottomBar())

        self._log_entries = 0

    def set_counter(self, name: str, value: int):
        card = self._cards.get(name)
        if card:
            card.set_value(value)

    def add_log(self, text: str, dot_color: str = C_TEAL):
        ts = QDateTime.currentDateTime().toString("hh:mm:ss")

        if self._log_entries == 0 and self._placeholder is not None:
            self._placeholder.setParent(None)
            self._placeholder.deleteLater()
            self._placeholder = None

        if self._log_entries > 0:
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(
                f"background:{C_BORDER}; max-height:1px; border:none;"
            )
            self.log_layout.insertWidget(self.log_layout.count() - 1, sep)

        item = LogItem(dot_color, text, ts)
        self.log_layout.insertWidget(self.log_layout.count() - 1, item)
        self._log_entries += 1
        self.log_count_lbl.setText(
            f"{self._log_entries} entr{'y' if self._log_entries == 1 else 'ies'}"
        )
        QTimer.singleShot(
            30,
            lambda: self.scroll.verticalScrollBar().setValue(
                self.scroll.verticalScrollBar().maximum()
            ),
        )

    def clear_logs(self):
        while self.log_layout.count() > 1:
            item = self.log_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._log_entries = 0
        self.log_count_lbl.setText("0 entries")
        self._placeholder = EmptyLogPlaceholder()
        self.log_layout.insertWidget(0, self._placeholder)

    def configure(self, mode="new"):
        now = QDateTime.currentDateTime().toString("hh:mm")
        self.clear_logs()
        if mode == "resumed":
            self.topbar.set_badge("Resumed session")
            self.status_lbl.setText(f"● Resumed — previous state restored · {now}")
        else:
            self.topbar.set_badge("Active session")
            self.status_lbl.setText(f"● New session started · {now}")
        for card in self._cards.values():
            card.set_value(0)
        self.card_graded.set_sub("Awaiting completion")