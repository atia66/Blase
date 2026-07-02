
import psycopg2
import os
import pandas as pd
from tkinter import Tk, filedialog
import os
import json 
from dotenv import load_dotenv
from benchmark.academic_report import build_report_image
load_dotenv()
DSN = os.getenv("DSN")
conn = psycopg2.connect(DSN)
conn.autocommit = True

def show_request():
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM "sheet_request";
        """,
    )
    results = cur.fetchall()
    cur.close()
    return results

def show_MCQ():
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM MCQ ;
        """,
    )
    results = cur.fetchall()
    cur.close()
    return results

def show_Student_Answer():
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM Student_Answer ;
        """,
    )
    results = cur.fetchall()
    cur.close()
    return results

def show_Model_Answer():
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM Model_Answer ;
        """,
    )
    results = cur.fetchall()
    cur.close()
    return results

def show_Complete():
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM Complete ;
        """,
    )
    results = cur.fetchall()
    cur.close()
    return results

def show_written():
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM written ;
        """,
    )
    results = cur.fetchall()
    cur.close()
    return results

def show_grads():
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM grads ;
        """,
    )
    results = cur.fetchall()
    cur.close()
    return results

def drop_request():
    cur = conn.cursor()
    try:
        
        cur.execute(
            """
            DROP 
            TABLE sheet_request ;
            """,
        )
        cur.close()
    except:
        pass

def drop_MCQ():
    cur = conn.cursor()
    try:
        cur.execute(
            """
            DROP 
            TABLE MCQ ;
            """,
        )
        cur.close()
    except:
        pass

def drop_Student_Answer():
    try:
        cur = conn.cursor()

        cur.execute(
            """
            DROP 
            TABLE Student_Answer ;
            """,
        )
        cur.close()
    except:
        pass

def drop_Model_Answer():
    try:
            
        cur = conn.cursor()

        cur.execute(
            """
            DROP 
            TABLE Model_Answer ;
            """,
        )
        cur.close()
    except:
        pass

def drop_Complete():
    try:
            
        cur = conn.cursor()

        cur.execute(
            """
            DROP 
            TABLE Complete ;
            """,
        )
        cur.close()
    except:
        pass

def drop_written():
    try:
        cur = conn.cursor()

        cur.execute(
            """
            DROP 
            TABLE written ;
            """,
        )
        cur.close()
    except:
        pass

def drop_grading():
    try:
            
        cur = conn.cursor()

        cur.execute(
            """
            DROP 
            TABLE grads ;
            """,
        )
        cur.close()
    except:
        pass

def show_grads(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM grads")
    results = cur.fetchall()
    cur.close()
    return results


def export_grads_to_csv(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = 'grads'
        )
    """)
    exists = cur.fetchone()[0]
    cur.close()

    if not exists:
        print("No grads table found — nothing to export.")
        return

    df = pd.read_sql_query("SELECT * FROM grads", conn)
    if df.empty:
        print("Grads table is empty — nothing to export.")
        return

    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory(title="Choose folder to save CSV")
    root.destroy()

    if not folder:
        print("Operation cancelled.")
        return

    report_df = df.rename(columns={
        "choice_score":          "mcq",
        "choice_total_score":    "max_mcq",
        "complete_score":        "complete",
        "complete_total_score":  "max_complete",
        "written_score":         "written",
        "written_total_score":   "max_written",
        "total_score":           "total",
        "total":                 "max_total",
    })

    build_report_image(report_df, f"{folder}/blase_report.png")

    file_path = os.path.join(folder, "grads.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} rows to {file_path}")


def drop_tables():
    drop_request()
    drop_MCQ()
    drop_Complete()
    drop_written()
    drop_Model_Answer()
    drop_Student_Answer()
    drop_grading()



