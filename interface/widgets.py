from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from interface.styles import (
    font, h_line, solid_btn, outline_btn,
    FONT_MAIN,
    C_BG_DEEP, C_BG_BASE, C_BG_CARD, C_BG_TOPBAR,
    C_BORDER, C_TEXT_PRI, C_TEXT_MUT, C_TEXT_DIM,
    C_TEAL,
)


class CounterCard(QWidget):
    def __init__(self, label, accent, sub="—", parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(148)
        self.setStyleSheet(f"""
            CounterCard {{
                background:{C_BG_CARD};
                border-radius:10px;
                border-top:3px solid {accent};
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(2)

        lbl = QLabel(label.upper())
        lbl.setFont(font(9, bold=True))
        lbl.setStyleSheet(f"color:{accent}; letter-spacing:2px; background:transparent;")
        lay.addWidget(lbl)
        lay.addSpacing(4)

        self.num = QLabel("0")
        self.num.setFont(QFont(FONT_MAIN, 44, QFont.Medium))
        self.num.setStyleSheet(f"color:{C_TEXT_PRI}; background:transparent;")
        lay.addWidget(self.num)

        self.sub = QLabel(sub)
        self.sub.setFont(font(10))
        self.sub.setStyleSheet(f"color:{C_TEXT_DIM}; background:transparent;")
        lay.addWidget(self.sub)

    def set_value(self, v):
        self.num.setText(str(v))

    def set_sub(self, s):
        self.sub.setText(s)


class LogItem(QWidget):
    def __init__(self, dot_color, text, time_str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 12, 0, 12)
        lay.setSpacing(14)
        lay.setAlignment(Qt.AlignTop)

        dot = QLabel("●")
        dot.setFont(font(8))
        dot.setStyleSheet(f"color:{dot_color}; background:transparent; padding-top:2px;")
        dot.setFixedWidth(12)
        lay.addWidget(dot)

        right = QVBoxLayout()
        right.setSpacing(2)

        msg = QLabel(text)
        msg.setFont(font(11))
        msg.setStyleSheet(f"color:{C_TEXT_MUT}; background:transparent;")
        msg.setWordWrap(True)
        right.addWidget(msg)

        ts = QLabel(time_str)
        ts.setFont(font(10))
        ts.setStyleSheet(f"color:{C_TEXT_DIM}; background:transparent;")
        right.addWidget(ts)

        lay.addLayout(right)


class EmptyLogPlaceholder(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setContentsMargins(0, 60, 0, 60)

        icon = QLabel("○")
        icon.setFont(font(28))
        icon.setStyleSheet(f"color:{C_BORDER}; background:transparent;")
        icon.setAlignment(Qt.AlignCenter)
        lay.addWidget(icon)
        lay.addSpacing(10)

        msg = QLabel("No log entries yet")
        msg.setFont(font(13))
        msg.setStyleSheet(f"color:{C_TEXT_DIM}; background:transparent;")
        msg.setAlignment(Qt.AlignCenter)
        lay.addWidget(msg)

        hint = QLabel("Events will appear here as the pipeline runs.")
        hint.setFont(font(10))
        hint.setStyleSheet(f"color:{C_BORDER}; background:transparent;")
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)


class TopBar(QWidget):
    def __init__(self, show_breadcrumb=False, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setStyleSheet(f"background:{C_BG_TOPBAR}; border-bottom:1px solid {C_BORDER};")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 0, 24, 0)

        logo = QLabel("BLASE")
        logo.setFont(font(14, bold=True))
        logo.setStyleSheet(f"color:{C_TEXT_PRI}; letter-spacing:0.5px; background:transparent;")
        lay.addWidget(logo)

        if show_breadcrumb:
            for txt, color, pad in [
                ("/",                C_TEXT_DIM, "4"),
                ("Sessions",         C_TEXT_DIM, "0"),
                ("›",                C_TEXT_DIM, "2"),
                ("Grading dashboard",C_TEXT_PRI,  "0"),
            ]:
                lbl = QLabel(txt)
                lbl.setFont(font(11 if txt not in ("/", "›") else 13))
                lbl.setStyleSheet(
                    f"color:{color}; background:transparent; padding:0 {pad}px;"
                )
                lay.addWidget(lbl)

        lay.addStretch()

        self.badge = QLabel("● System ready")
        self.badge.setFont(font(10))
        self.badge.setStyleSheet(f"""
            color:{C_TEAL};
            background:rgba(52,211,153,0.08);
            border:0.5px solid rgba(52,211,153,0.25);
            border-radius:20px; padding:3px 12px;
        """)
        lay.addWidget(self.badge)

    def set_badge(self, text):
        self.badge.setText(f"● {text}")


class BottomBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setStyleSheet(f"background:{C_BG_DEEP}; border-top:1px solid {C_BORDER};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 0, 24, 0)

        right = QLabel("● Connected")
        right.setFont(font(10))
        right.setStyleSheet(f"color:{C_TEXT_DIM}; background:transparent;")
        lay.addWidget(right)