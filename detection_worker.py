import psycopg2
import select
from sheet_detection.Worker import detection_worker
import time
from metrics import record_request

def postgres_connection(DSN):
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    return conn

def show_request(conn, job_id):
    cur = conn.cursor()

    cur.execute(
        """
        SELECT Request
        FROM sheet_request
        WHERE Job_Id = %s;
        """,
        (job_id,)
    )

    result = cur.fetchone()

    cur.close()

    if result is None:
        return None

    return result[0]

def detection_consumer(DSN, log_queue=None):

    def log(msg, color=None):
        if log_queue:
            log_queue.put((msg, color) if color else msg)
        else:
            print(msg)

    def counter(name, value):
        if log_queue:
            log_queue.put(("__counter__", name, value))

    conn = postgres_connection(DSN)
    cur  = conn.cursor()

    cur.execute("LISTEN Requests;")
    log("DETECTION WORKER: ready, listening on Requests")

    requests_count = 0
    students_count = 0
    model_count    = 0

    while True:
        if select.select([conn], [], [], 5) == ([], [], []):
            continue

        conn.poll()

        while conn.notifies:
            t0 = time.time()

            notify  = conn.notifies.pop(0)
            job_id  = notify.payload


            img_bytes = show_request(conn, job_id)

            if img_bytes is None:
                log(f"DETECTION WORKER: job {job_id} not found — skipping", "#FCA5A5")
                continue
            try:
                t0 = time.time()       
                sheet_type=detection_worker(job_id, img_bytes)
                record_request(
                        "detection_worker",
                        t0,)
                requests_count += 1
                counter("requests", requests_count)
                log(f"DETECTION WORKER: job {job_id} processed ✓", "#34D399")
                if sheet_type == "Student_Answer":
                    students_count += 1
                    counter("students", students_count)
                    log(f"DETECTION WORKER: Recived Student answer sheet ✓", "#34D399")
                elif sheet_type == "Model_Answer":
                    model_count += 1
                    counter("model", model_count)
                    log(f"DETECTION WORKER: Recived Model answer sheet ✓", "#34D399")
            except Exception as e:
                log(f"DETECTION WORKER: job {job_id} FAILED — {e}", "#FCA5A5")

def show_table(conn):
    cur = conn.cursor()

    cur.execute(
        """
        SELECT * FROM sheet_request;
        """
    )

    results = cur.fetchall()

    cur.close()

    return results


