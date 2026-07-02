from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from interface.styles import (
    font, h_line, solid_btn, outline_btn,
    FONT_MAIN,
    C_BG_BASE, C_TEXT_PRI, C_TEXT_DIM, C_TEAL,
)
from interface.widgets import TopBar, BottomBar


class WelcomeScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{C_BG_BASE};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.topbar = TopBar(show_breadcrumb=False)
        root.addWidget(self.topbar)

        center_wrap = QWidget()
        center_wrap.setStyleSheet(f"background:{C_BG_BASE};")
        center_lay = QVBoxLayout(center_wrap)
        center_lay.setAlignment(Qt.AlignCenter)
        center_lay.setSpacing(0)

        title = QLabel("BLASE")
        title.setFont(QFont(FONT_MAIN, 64, QFont.Medium))
        title.setStyleSheet(
            f"color:{C_TEXT_PRI}; letter-spacing:-2px; background:transparent;"
        )
        title.setAlignment(Qt.AlignCenter)
        center_lay.addWidget(title)

        center_lay.addSpacing(24)
        center_lay.addWidget(h_line())
        center_lay.addSpacing(16)

        meta = QLabel("Version 1.0")
        meta.setFont(font(10))
        meta.setStyleSheet(f"color:{C_TEXT_DIM}; background:transparent;")
        meta.setAlignment(Qt.AlignCenter)
        center_lay.addWidget(meta)

        center_lay.addSpacing(40)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.setAlignment(Qt.AlignCenter)

        self.continue_btn = outline_btn(
            "Recover last session", C_TEAL, "rgba(52,211,153,0.35)",
            min_w=180, min_h=44,
        )
        btn_row.addWidget(self.continue_btn)

        self.start_btn = solid_btn(
            "New session", "#1D5FCC", C_TEXT_PRI, "#2471E8",
            min_w=180, min_h=44,
        )
        btn_row.addWidget(self.start_btn)

        center_lay.addLayout(btn_row)
        center_lay.addSpacing(10)

        hint = QLabel("Start a new grading session.")
        hint.setFont(font(10))
        hint.setStyleSheet(f"color:{C_TEXT_DIM}; background:transparent;")
        hint.setAlignment(Qt.AlignCenter)
        center_lay.addWidget(hint)

        root.addWidget(center_wrap, 1)
        root.addWidget(BottomBar())