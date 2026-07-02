# BLASE Grading & Question Generation System

## Overview

This project implements an event-driven exam grading system and a separate question generation service. It uses independent workers connected through message queues, which improves scalability and reduces end-to-end latency compared to traditional batch processing.

## How to run

1. Run the grading system:
   - Start the main application:
     ```bash
     python main.py
     ```
   - After the system is running, send a grading request using:
     ```bash
     python exam_client.py
     ```
   - `exam_client.py` sends an example answer sheet image to the grading API.

2. Run question generation:
   - Start the question generation server first:
     ```bash
     python question_generation/server.py
     ```
   - Then send a PDF request using:
     ```bash
     python client.py
     ```
   - If the client and server ports differ, update `client.py` to match the server host and port.

## Why Event-Driven Architecture

Before explaining the system, it is important to clarify why we chose Event-Driven Architecture over Batch Processing.

Batch Processing waits until sheets are collected before the pipeline begins, whether sequentially or in parallel. That means there is an initial delay before processing starts, which increases end-to-end latency.

Event-Driven Architecture breaks the system into independent workers. Each worker communicates with the next stage through a message queue. Once a worker completes its task, it publishes a message to the next queue, and the next worker consumes it when ready.

For example, when a user uploads a sheet, the server publishes a message to the Detection Worker queue. After detection completes, the next stage begins, and the process continues until grading is finished.

Even if Batch Processing had no idle time, Event-Driven Architecture remains more scalable for this project because each stage is independent. We can add more workers to a bottleneck stage without affecting the rest of the pipeline.

## System Workers

The system includes five main workers:

- Detection Worker
- Complete Worker
- Written Worker
- Grading Worker
- Question Generation Worker (separate from grading)

### Detection Worker

The Detection Worker extracts the main data from an answer sheet, such as:

- QR code
- Student ID
- Form ID
- Bubble sheet containers

It uses color segmentation with the HSV color space.

The sheet also includes alignment markers to ensure stable extraction even if the image is tilted or captured from an angle. Corner marks are used to apply homography and perspective correction to flatten the sheet, while timing marks help determine orientation and maintain element alignment.

### Complete Worker

The Complete Worker grades complete-answer questions using a two-stage OCR pipeline.

- Stage 1: text detection using CRAFT to locate text regions.
- Stage 2: text recognition using CRNN to convert text regions into text.

### Written Worker

The Written Worker relies on a vision-language model (Gemma).

This model can understand both image content and context, and it generates answers in LaTeX format. It is therefore better at handling mathematical equations, symbols, and code than traditional OCR.

### Why Complete and Written are different

The OCR pipeline works well for ordinary text, but it struggles with equations, math symbols, and code.

The vision-language model is better suited for this type of content.

### Grading Worker

The grading stage evaluates student answers against model answers.

OCR output may contain recognition errors, so simple string matching is not enough.

Embedding similarity was considered but rejected, since embeddings measure semantic similarity rather than answer correctness.

For example, if the model answer is "Machine Learning" and the student writes "AI" or "Deep Learning," the sentences may be semantically close but not necessarily the required answer.

Therefore, the system uses a Two-Agent Grading System.

The agent compares the student answer with the model answer and determines correctness based on question context rather than just word similarity.

### Difference between Complete Agent and Written Agent

In the Complete Worker, all answers are graded in a single request because the responses are short and token usage is low.

In the Written Worker, each question is graded in a separate request. Combining many written questions in one prompt increases hallucination risk and answer mixing, so separating them improves grading accuracy.

### Question Generation Worker

The Question Generation Worker is the final service.

It takes lecture material, divides it into chunks of around nine slides, extracts key points from each chunk, and generates three types of questions:

- MCQ
- Complete
- Written

## Project Structure

- `.env`
- `.gitignore`
- `main.py` — starts the main GUI and worker processes
- `exam_client.py` — example grading request sender
- `client.py` — example PDF request sender for question generation
- `complete_worker.py` — complete-answer OCR worker
- `detection_worker.py` — sheet detection worker
- `written_worker.py` — written-answer worker
- `grading_worker.py` — grading worker
- `tables.py` — database table utilities and export functions
- `metrics.py` — metrics recording and monitoring helpers
- `processing.py` — general processing utilities

Directories:

- `Grading/`
  - `complete_graph.py`
  - `written_graph.py`
  - `MCQ.py`
  - `preprocess.py`
  - `tools.py`
  - `prompts/written.txt`
- `ground_truth/`
  - reference answers and grading ground truth
- `interface/` — PySide6 GUI screens and styles
- `Machine/` — main request server and producer logic
- `question_generation/` — PDF chunking and question generation service
- `sheet_detection/` — sheet detection and extraction utilities
- `Text_detection/` — CRAFT text detection implementation
- `Text_recognition/` — CRNN text recognition implementation
- `report_output/` — generated reports and visuals

## Notes

- `main.py` launches the full grading pipeline including detection, complete, written, and grading workers.
- `exam_client.py` is the recommended request example for the grading API.
- `question_generation/server.py` must run before sending requests from `client.py`.
- The project uses multiple services and may require dependencies from `requirements.txt`.

