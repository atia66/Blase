from google import genai
import cv2
import psycopg2
import select 
from sheet_detection.producer import answers_worker
import numpy as np
from google import genai
from google.genai import types
import time
import os
from dotenv import load_dotenv
import time
from metrics import record_request, log_usage
import random

load_dotenv()
USAGE_PATH = "./matrics/data.json"


def image_decode(data):
    try:
        if isinstance(data, memoryview):
            data = bytes(data)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("decode error:", e)
        return None

def show_data(job_id, conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, image, Sheet_Type
        FROM written
        WHERE Job_Id = %s 
        AND status = 'PENDING'
        LIMIT 1
        FOR UPDATE SKIP LOCKED;
        """, (job_id,)
    )
    row = cur.fetchone()
    cur.close()
    if row is None:
        return None, None, None
    id, img_bytes, Sheet_Type = row
    return id, img_bytes, Sheet_Type

def update_status(id, conn):
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE written
        SET status = 'PROCESSING'
        WHERE id = %s
        """, (id,)
    )
    conn.commit()
    cur.close()

def update_status2(id, conn):
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE written
        SET status = 'Done'
        WHERE id = %s
        """, (id,)
    )
    conn.commit()
    cur.close()

def written_worker(img, max_retries=3):
    if img is None:
        raise ValueError("image_decode returned None — corrupt or invalid image bytes")

    client = genai.Client(api_key=os.getenv("api_key"))

    _, buffer = cv2.imencode(".jpg", img)
    img_bytes = buffer.tobytes()

    if len(img_bytes) > 15 * 1024 * 1024:
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        img_bytes = buffer.tobytes()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=os.getenv("model"),
                contents=[
                    "Extract all text, handwritten and equations and return only LaTeX.",
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/jpeg"
                    )
                ]
            )
            usage = {
                "tokens_input": response.usage_metadata.prompt_token_count,
                "tokens_output": response.usage_metadata.candidates_token_count,
                "api_calls": 1
            }
            return response.text, usage

        except Exception as e:
            err = str(e)
            if ("500" in err or "INTERNAL" in err) and attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"WRITTEN WORKER: Gemini 500, retry {attempt + 1}/{max_retries} in {wait:.1f}s")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Gemini 500 after {max_retries} retries")

def postgres_connection(DSN):
    conn = psycopg2.connect(DSN)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    conn.autocommit = True
    return conn

def written_consumer(DSN, log_queue=None):

    def log(msg, color=None):
        if log_queue:
            log_queue.put((msg, color) if color else msg)
        else:
            print(msg)

    def counter(name, value):
        if log_queue:
            log_queue.put(("__counter__", name, value))

    conn = postgres_connection(DSN)
    cur = conn.cursor()
    cur.execute("LISTEN written_channel;")
    log("WRITTEN WORKER: ready, listening on written_channel")

    while True:
        if select.select([conn], [], [], 5) == ([], [], []):
            continue

        conn.poll()

        while conn.notifies:
            notify = conn.notifies.pop(0)
            job_id = notify.payload

            try:
                id, img, Sheet_Type = show_data(job_id, conn)
                if id is None:
                    log(f"WRITTEN WORKER: job {job_id} — no PENDING row found (skipping)", "#94A3B8")
                    continue
                update_status(id, conn)
                t0 = time.time()
                img = image_decode(img)
                answer, usage = written_worker(img)
                record_request("written_worker", t0, tokens=usage)
                if Sheet_Type == "Student_Answer":
                    model = answers_worker("Student_Answer")
                    model.update_answer_written(job_id, answer)
                    log(f"WRITTEN WORKER: student written answer extracted — job {job_id}", "#4A90E2")
                else:
                    model = answers_worker("Model_Answer")
                    model.update_answer_written(job_id, answer)
                    log(f"WRITTEN WORKER: model written answer extracted — job {job_id}", "#F5A623")

                update_status2(id, conn)
                log(f"WRITTEN WORKER: job {job_id} done ✓", "#34D399")
                time.sleep(15)

            except Exception as e:
                log(f"WRITTEN WORKER: job {job_id} FAILED — {e}", "#FCA5A5")