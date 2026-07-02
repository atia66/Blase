import json
import uuid
import psycopg2
DSN = "dbname=postgres user=postgres password=postgres123 host=localhost"

def postgres_connection(DSN):
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    return conn


def create_request_table(conn):
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sheet_request (
            id SERIAL PRIMARY KEY,
            Job_Id UUID,
            Request TEXT,
            status TEXT DEFAULT 'PENDING'
        );
    """)

    cur.close()


def drop_request(conn):
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS sheet_request;")

    cur.close()
conn=postgres_connection(DSN)
drop_request(conn)

def insert_requests(conn, job_id, request_data):
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO sheet_request (Job_Id, Request)
        VALUES (%s, %s)
        """,
        (job_id, request_data)
    )

    cur.close()


def request_producer(bytes_img):
    conn = postgres_connection(DSN)

    create_request_table(conn)

    job_id = str(uuid.uuid4())

    insert_requests(conn, job_id, bytes_img)

    cur = conn.cursor()

    cur.execute(
        "NOTIFY Requests, %s;",
        (job_id,)
    )

    cur.close()

    conn.close()

    print(f"Produced Job: {job_id}")






