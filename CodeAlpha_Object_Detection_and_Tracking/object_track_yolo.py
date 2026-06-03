"""
TASK 4: Object Detection and Tracking

Uses YOLOv8 for detection + a simple SORT-style tracker.

Requirements (install once):
    pip install ultralytics opencv-python numpy

Run:
    python object_track_yolo.py               # uses webcam
    python object_track_yolo.py --file vid.mp4  # uses a video file
"""

import argparse   # lets us pass command-line flags (like --file)
import time       # for measuring time / FPS
import cv2                          # OpenCV: reads frames, draws boxes, shows window
import numpy as np                  # NumPy: fast maths on arrays
from ultralytics import YOLO        # YOLOv8 from Ultralytics (wraps the model nicely)


# Simple SORT-style tracker (IoU matching, no deep features)

def iou(box_a, box_b):
    """
    Compute Intersection-over-Union between two bounding boxes.

    A box is [x1, y1, x2, y2] (top-left corner, bottom-right corner).
    IoU = overlap_area / union_area.
    A value close to 1 means the boxes almost perfectly overlap.
    A value close to 0 means they barely touch.
    """
    # Find the top-left corner of the intersection rectangle
    x_left   = max(box_a[0], box_b[0])
    y_top    = max(box_a[1], box_b[1])

    # Find the bottom-right corner of the intersection rectangle
    x_right  = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    # If the boxes don't overlap at all, IoU is 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Area of the overlapping rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Area of each individual box
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Union = sum of both areas minus the counted-twice intersection
    union_area = area_a + area_b - intersection_area

    return intersection_area / union_area  # final IoU score


class SimpleTracker:
    """
    A lightweight object tracker based on IoU matching.

    How it works:
      1. For each new frame we get a list of detected bounding boxes.
      2. We try to match each detection to an existing tracked object
         by finding which existing track has the highest IoU overlap.
      3. Matched tracks get their box updated and their 'miss' counter reset.
      4. Unmatched tracks get a 'miss' count increase; after too many misses
         the track is deleted (the object probably left the frame).
      5. Unmatched detections start a brand-new track with a new ID.
    """

    def __init__(self, iou_threshold=0.3, max_misses=5):
        """
        iou_threshold : minimum IoU needed to call two boxes 'the same object'
        max_misses    : how many consecutive frames without a match before we
                        delete a track
        """
        self.tracks       = {}   # dict  { track_id : info_dict }
        self.next_id      = 1    # auto-increment ID for new tracks
        self.iou_threshold = iou_threshold
        self.max_misses    = max_misses

    def update(self, detections):
        """
        Match detections (list of [x1,y1,x2,y2,conf,class_id]) to existing
        tracks and return a list of (box, track_id, class_id, conf).
        """

        # try to match every detection to an existing track 
        matched_track_ids   = set()   # tracks that got a detection this frame
        matched_det_indices = set()   # detections that were matched

        track_ids   = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["box"] for tid in track_ids]

        for det_idx, det in enumerate(detections):
            det_box    = det[:4]
            best_iou   = 0
            best_track = None

            for trk_idx, trk_box in enumerate(track_boxes):
                score = iou(det_box, trk_box)
                if score > best_iou:
                    best_iou   = score
                    best_track = track_ids[trk_idx]

            if best_iou >= self.iou_threshold and best_track not in matched_track_ids:
                self.tracks[best_track]["box"]      = det_box
                self.tracks[best_track]["conf"]     = det[4]
                self.tracks[best_track]["class_id"] = int(det[5])
                self.tracks[best_track]["misses"]   = 0

                matched_track_ids.add(best_track)
                matched_det_indices.add(det_idx)

        # increase miss counter for unmatched tracks 
        for tid in track_ids:
            if tid not in matched_track_ids:
                self.tracks[tid]["misses"] += 1

        # delete tracks that have been missing too long 
        self.tracks = {
            tid: info
            for tid, info in self.tracks.items()
            if info["misses"] <= self.max_misses
        }

        # Step D: create new tracks for unmatched detections 
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                self.tracks[self.next_id] = {
                    "box"      : det[:4],
                    "conf"     : det[4],
                    "class_id" : int(det[5]),
                    "misses"   : 0,
                }
                self.next_id += 1

        # return all currently active tracks 
        results = []
        for tid, info in self.tracks.items():
            results.append((info["box"], tid, info["class_id"], info["conf"]))
        return results


# Drawing helpers

PALETTE = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]

def color_for_id(track_id):
    """Pick a consistent BGR colour for a given track ID."""
    return PALETTE[track_id % len(PALETTE)]


def draw_tracked_object(frame, box, track_id, label, conf):
    """
    Draw a bounding box + label on the frame for one tracked object.
    """
    x1, y1, x2, y2 = map(int, box)
    colour = color_for_id(track_id)

    # Draw bounding box rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Build the label text
    text = f"ID:{track_id} {label} {conf:.2f}"

    # Measure text size so we can draw a background behind it
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    )

    # Filled rectangle as text background (for readability)
    cv2.rectangle(
        frame,
        (x1, y1 - text_h - baseline - 4),
        (x1 + text_w, y1),
        colour,
        cv2.FILLED,
    )

    # White text on top of the coloured background
    cv2.putText(
        frame, text,
        (x1, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_fps(frame, fps):
    """Overlay the current FPS in the top-left corner."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 0), 2, cv2.LINE_AA,
    )


# Main pipeline

def run(source, conf_threshold=0.4):
    """
    Main function: opens video, runs YOLO detection, tracks objects, displays result.
    """

    # Load the YOLOv8 nano model (downloads ~6 MB on first run)
    print("[INFO] Loading YOLOv8n model ...")
    model = YOLO("yolov8n.pt")
    class_names = model.names   # {0: 'person', 1: 'bicycle', ...}

    # Open webcam (int 0) or video file (string path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    print(f"[INFO] Video source opened: {source}")
    print("[INFO] Press 'q' to quit.")

    # Create one tracker instance that lives for the whole video
    tracker  = SimpleTracker(iou_threshold=0.3, max_misses=5)
    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        # Run YOLO on the frame 
        results   = model(frame, verbose=False)
        boxes_data = results[0].boxes

        # Convert YOLO detections to [x1,y1,x2,y2,conf,class_id] list 
        detections = []
        for box in boxes_data:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue                         # skip weak detections
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf, class_id])

        # Update tracker and get (box, id, class, conf) per object 
        tracked_objects = tracker.update(detections)

        # Draw each tracked object on the frame 
        for (box, track_id, class_id, conf) in tracked_objects:
            label = class_names.get(class_id, "unknown")
            draw_tracked_object(frame, box, track_id, label, conf)

        # Compute FPS and overlay it 
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-9)
        prev_time = current_time
        draw_fps(frame, fps)

        # Show the annotated frame 
        cv2.imshow("YOLOv8 + Tracker  (press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection + Tracking")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a video file (omit for webcam).")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Minimum detection confidence (default 0.4).")
    args = parser.parse_args()

    video_source = args.file if args.file else 0
    run(source=video_source, conf_threshold=args.conf)
