import select
import json
import traceback
import psycopg2
from psycopg2.extras import DictCursor

from Grading.complete_graph import complete_producer
from Grading.written_graph import written_producer
from Grading.MCQ import MCQ_worker
import time
from metrics import record_request, log_usage
import os
from dotenv import load_dotenv
load_dotenv()
DSN = os.getenv("DSN")
USAGE_PATH = "./matrics/data.json"
import requests


def postgres_connection(DSN, autocommit=True):
    conn = psycopg2.connect(DSN)
    conn.autocommit = autocommit
    return conn


def grading_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS grads (
            id                   SERIAL PRIMARY KEY,
            student_id           BIGINT,
            instructor_id        INTEGER,
            course_id            TEXT,
            choice_score         INTEGER,
            choice_total_score   INTEGER,
            complete_score       INTEGER,
            complete_total_score INTEGER,
            written_score        INTEGER,
            written_total_score  INTEGER,
            total_score          INTEGER,
            total                INTEGER,
            graded_at            TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()

def fetch_job_data(job_id, conn):
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            "SELECT id ,exam_form, instructor_id, course_id, student_id, "
            "       choice_answers, complete_answers, written_answers "
            "FROM Student_Answer WHERE job_id = %s",
            (job_id,)
        )
        sa_row = cur.fetchone()

        if sa_row is None:
            print(f"[{job_id}] LOOKUP FAIL: no Student_Answer row with this job_id")
            conn.rollback()
            return None

        cur.execute(
            "SELECT choice_answers ,exam_form, complete_answers, written_answers "
            "FROM Model_Answer WHERE instructor_id = %s AND course_id = %s and exam_form =%s" ,
            (sa_row["instructor_id"], sa_row["course_id"],sa_row["exam_form"])
        )
        ma_row = cur.fetchone()

        if ma_row is None:
            print(f"[{job_id}] LOOKUP FAIL: no matching Model_Answer for "
                  f"instructor_id={sa_row['instructor_id']} course_id={sa_row['course_id']}")
            conn.rollback()
            return None

        print(f"[{job_id}] LOOKUP OK: row_id={sa_row['id']} "
              f"student_id={sa_row['student_id']} "
              f"instructor_id={sa_row['instructor_id']} course_id={sa_row['course_id']}")

        conn.rollback()
        return {
            "row_id":        sa_row["id"],
            "student_id":    sa_row["student_id"],
            "instructor_id": sa_row["instructor_id"],
            "course_id":     sa_row["course_id"],
            "s_choice":      sa_row["choice_answers"],
            "s_complete":    sa_row["complete_answers"],
            "s_written":     sa_row["written_answers"],
            "m_choice":      ma_row["choice_answers"],
            "m_complete":    ma_row["complete_answers"],
            "m_written":     ma_row["written_answers"],
        }

def insert_grade(conn, data, scores):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO grads (
                student_id, instructor_id, course_id,
                choice_score, choice_total_score,
                complete_score, complete_total_score,
                written_score, written_total_score,
                total_score, total
            )
            VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            data["student_id"], data["instructor_id"], data["course_id"],
            scores["choice_score"], scores["choice_total"],
            scores["complete_score"], scores["complete_total"],
            scores["written_score"], scores["written_total"],
            scores["total_score"], scores["total"],
        ))
    conn.commit()
    print(f"[grades] inserted grade row for student_id={data['student_id']} "
          f"course_id={data['course_id']}")


def update_status(row_id, status, conn):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE Student_Answer SET status = %s WHERE id = %s",
            (status, row_id)
        )
    conn.commit()
    print(f"[row_id={row_id}] status -> {status}")


def process_job(job_id, work_conn):
    print(f"[{job_id}] processing started")

    data = fetch_job_data(job_id, work_conn)

    if data is None:
        print(f"[{job_id}] No data found — skipping")
        return None

    print(f"[{job_id}] student={data['student_id']} course={data['course_id']}")

    update_status(data["row_id"], "PROCESSING", work_conn)

    choice_result, choice_score, choice_total = MCQ_worker(
        data["s_choice"], data["m_choice"]
    )
    print(f"[{job_id}] MCQ graded: {choice_score}/{choice_total}")

    complete_result, complete_score, complete_total, usage_complete = complete_producer(
        data["m_complete"], data["s_complete"]
    )
    print(f"[{job_id}] complete-answers graded: {complete_score}/{complete_total}")

    written_result, written_score, written_total, usage_written = written_producer(
        data["m_written"], data["s_written"]
    )
    print(f"[{job_id}] written-answers graded: {written_score}/{written_total}")

    usage = {"usage_written": usage_written, "usage_complete": usage_complete}

    scores = {
        "choice_score": choice_score,
        "choice_total": choice_total,
        "complete_score": complete_score,
        "complete_total": complete_total,
        "written_score": written_score,
        "written_total": written_total,
        "total_score": choice_score + complete_score + written_score,
        "total": choice_total + complete_total + written_total,
    }

    insert_grade(work_conn, data, scores)

    base_url = "http://localhost:8080/api/grades"

    payload = {
        "choice_score": int(scores["choice_score"]),
        "choice_total_score": int(scores["choice_total"]),

        "complete_score": int(scores["complete_score"]),
        "complete_total_score": int(scores["complete_total"]),

        "written_score": int(scores["written_score"]),
        "written_total_score": int(scores["written_total"]),

        "total": int(scores["total_score"]),
        "total_score": int(scores["total"]),

        "student_id": data["student_id"],
        "course_id": data["course_id"],
        "instructor_id": data["instructor_id"],
    }

    headers = {
        "X-API-Key": "blase",
    }

    try:
        response = requests.post(base_url, headers=headers, json=payload, timeout=10)
        print(f"[{job_id}] POST /api/grades -> {response.status_code}")
    except requests.RequestException as e:
        print(f"[{job_id}] WARNING: POST /api/grades failed: {e}")

    os.makedirs("./ground_truth/predicted/grading", exist_ok=True)
    answers = {
        "studentId": data["student_id"],
        "mcq": choice_result,
        "complete": complete_result["results"],
        "written": written_result["results"],
    }
    out_path = f"./ground_truth/predicted/grading/grading_{data['student_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
    print(f"[{job_id}] wrote {out_path}")

    update_status(data["row_id"], "DONE", work_conn)
    print(f"[{job_id}] done")
    return usage


def grading_consumer(DSN, log_queue=None):

    def log(msg, color=None):
        if log_queue:
            log_queue.put((msg, color) if color else msg)
        else:
            print(msg)

    def counter(name, value):
        if log_queue:
            log_queue.put(("__counter__", name, value))

    listen_conn = postgres_connection(DSN, autocommit=True)
    work_conn = postgres_connection(DSN, autocommit=False)

    grading_table(work_conn)

    with listen_conn.cursor() as cur:
        cur.execute("LISTEN grading_channel;")

    log("GRADING WORKER: listening on grading_channel", "#60A5FA")

    graded_count = 0

    while True:
        ready = select.select([listen_conn], [], [], 5)
        if ready == ([], [], []):
            continue

        listen_conn.poll()

        while listen_conn.notifies:
            notify = listen_conn.notifies.pop(0)

            try:
                payload = json.loads(notify.payload)
                job_id = payload["job_id"]
            except (json.JSONDecodeError, KeyError):
                job_id = notify.payload

            log(f"GRADING WORKER: received notify for job {job_id}", "#93C5FD")

            try:
                t0 = time.time()
                usage = process_job(job_id, work_conn)

                if usage is None:
                    # process_job returned early (no data found) — nothing was
                    # graded, so don't count it as a success. The stale-
                    # transaction fix in fetch_job_data means this should now
                    # only happen for genuinely missing/mismatched jobs.
                    log(f"GRADING WORKER: job {job_id} skipped (no matching data)", "#FBBF24")
                    continue

                record_request(
                    "Grading_worker",
                    t0,
                    tokens=usage)
                graded_count += 1
                counter("graded", graded_count)
                log(f"GRADING WORKER: job {job_id} graded ✓", "#34D399")

            except Exception:
                work_conn.rollback()
                update_status_safe(job_id, "FAILED", work_conn)
                log(f"GRADING WORKER: job {job_id} FAILED", "#FCA5A5")
                traceback.print_exc()

def update_status_safe(job_id, status, conn):
    """Fallback status update by job_id when we don't have row_id."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE Student_Answer SET status = %s WHERE job_id = %s",
                (status, job_id)
            )
        conn.commit()
        print(f"[{job_id}] status -> {status} (via job_id fallback)")
    except Exception:
        conn.rollback()
        print(f"[{job_id}] status update FAILED, rolled back")