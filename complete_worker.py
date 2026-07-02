import psycopg2
from Text_detection.TD_worker import run_craft , copyStateDict
from Text_detection.craft import CRAFT
import torch
import cv2
from Text_recognition.demo import load_model, extract_text_from_image
from Text_recognition import utils
from Text_recognition import dataset
from PIL import Image
import select
from sheet_detection.producer import answers_worker
import numpy as np
import time
from metrics import record_request

class ModelState:
    model = None
    converter = None
    transformer = None

def image_decode(data):
    try:
        if isinstance(data, memoryview):
            data = bytes(data)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("decode error:", e)
        return None

def show_data(job_id,conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id,image,Sheet_Type
        FROM Complete
        WHERE Job_Id =%s 
        AND status = 'PENDING'
        LIMIT 1
        FOR UPDATE SKIP LOCKED;
        """,(job_id,)
    )
    id,img_bytes,Sheet_Type = cur.fetchone()
    cur.close()
    return id,img_bytes,Sheet_Type

def update_status(id,conn):
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE Complete
        SET status='PROCESSING'
        WHERE id =%s 

        """,(id,)
    )
    conn.commit()
    cur.close()

def update_status2(id,conn):
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE Complete
        SET status='Done'
        WHERE id =%s 

        """,(id,)
    )
    cur.close()

def postgres_connection(DSN):
    conn = psycopg2.connect(DSN)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    conn.autocommit = True
    return conn

def CRAFT_define():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./Text_detection/weights/craft_mlt_25k.pth"
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(model_path, map_location=device)))
    net.eval()
    return net

def CRNN_define(model_state:ModelState):
    
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    model_state.model = load_model('./Text_recognition/data/crnn.pth')
    model_state.converter = utils.strLabelConverter(alphabet)
    model_state.transformer = dataset.resizeNormalize((100, 32))
    return model_state

def complete_extraction(net,img,model_state:ModelState):
    crops=run_craft(net,img)
    results=""
    for crop in crops :
        crop=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
        crop = Image.fromarray(crop)
        text = extract_text_from_image(crop,model=model_state.model,converter=model_state.converter,transformer=model_state.transformer)
        results += text
    return results

def complete_consumer(DSN, log_queue=None):

    def log(msg, color=None):
        if log_queue:
            log_queue.put((msg, color) if color else msg)
        else:
            print(msg)

    conn = postgres_connection(DSN)
    net  = CRAFT_define()
    model_state = ModelState()
    model_state = CRNN_define(model_state)
    cur = conn.cursor()

    cur.execute("LISTEN Complete_channel;")

    while True:
        if select.select([conn], [], [], 5) == ([], [], []):
            continue

        conn.poll()

        while conn.notifies:
            notify = conn.notifies.pop(0)
            job_id = notify.payload


            try:
                t0 = time.time()       
                id, img, Sheet_Type = show_data(job_id, conn)
                update_status(id, conn)
                img    = image_decode(img)
                answer = complete_extraction(net, img, model_state)
                if Sheet_Type == "Student_Answer":
                    model = answers_worker("Student_Answer")
                    model.update_answer_choice(job_id, answer)
                    log(f"COMPLETE WORKER: student complete answer extracted — job {job_id}", "#4A90E2")
                else:
                    model = answers_worker("Model_Answer")
                    model.update_answer_choice(job_id, answer)
                    log(f"COMPLETE WORKER: Model complete answer extracted — job {job_id}", "#4A90E2")

                update_status2(id, conn)
                record_request(
                        "complete_worker",
                        t0,)

            except Exception as e:
                log(f"COMPLETE WORKER: job {job_id} FAILED — {e}", "#FCA5A5")