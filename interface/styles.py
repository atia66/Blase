from PySide6.QtWidgets import QPushButton, QFrame
from PySide6.QtGui import QFont

C_BG_DEEP   = "#081428"
C_BG_BASE   = "#0D2147"
C_BG_CARD   = "#112254"
C_BG_TOPBAR = "#0B1D3A"
C_BORDER    = "#1A3260"
C_TEXT_PRI  = "#E8F0FE"
C_TEXT_MUT  = "#8FA3BF"
C_TEXT_DIM  = "#3A567A"
C_BLUE      = "#4A90E2"
C_PURPLE    = "#9B7FE8"
C_AMBER     = "#F5A623"
C_TEAL      = "#34D399"
C_RED_BG    = "#7F1D1D"
C_RED_FG    = "#FCA5A5"
C_EXPBTN_BG = "#1A3260"
C_EXPBTN_FG = "#93C5FD"

FONT_MAIN = "Roboto"


def font(size, bold=False):
    f = QFont(FONT_MAIN, size)
    if bold:
        f.setWeight(QFont.Medium)
    return f


def h_line():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"background:{C_BORDER}; max-height:1px; border:none;")
    return f


def solid_btn(text, bg, fg, hover, min_w=160, min_h=42, fsize=12):
    b = QPushButton(text)
    b.setFont(font(fsize, bold=True))
    b.setMinimumSize(min_w, min_h)
    b.setCursor(b.cursor())
    b.setStyleSheet(f"""
        QPushButton {{
            background:{bg}; color:{fg};
            border:none; border-radius:6px;
            padding:8px 20px; letter-spacing:0.5px;
        }}
        QPushButton:hover {{ background:{hover}; }}
        QPushButton:pressed {{ background:{bg}; opacity:0.7; }}
    """)
    return b


def outline_btn(text, fg, border, min_w=160, min_h=42, fsize=12):
    b = QPushButton(text)
    b.setFont(font(fsize, bold=True))
    b.setMinimumSize(min_w, min_h)
    b.setStyleSheet(f"""
        QPushButton {{
            background:transparent; color:{fg};
            border:1px solid {border}; border-radius:6px;
            padding:8px 20px; letter-spacing:0.5px;
        }}
        QPushButton:hover {{ background:rgba(52,211,153,0.08); }}
        QPushButton:pressed {{ background:rgba(52,211,153,0.15); }}
    """)
    return b