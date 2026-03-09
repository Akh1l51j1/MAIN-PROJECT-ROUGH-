from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from ...models import Incident, Vehicle, VehicleDetection, SearchLog
import cv2
import numpy as np
import torch
import easyocr
from ultralytics import YOLO
from difflib import SequenceMatcher
import time
import threading
from queue import Queue
from collections import Counter

# ==========================================
# 🔧 HARDCODED CONFIGURATION (Backend Only)
# ==========================================
INCIDENT_VIDEO_PATH = "/kaggle/input/road-surveillance-test/accident.mp4"

# Use these feeds for tracking vehicles
CAMERA_FEEDS = {
    'cam1': {
        'path': "/kaggle/input/road-surveillance-test/cam2.mp4",
        'location': 'Main Street Intersection'
    },
    'cam2': {
        'path': "/kaggle/input/road-surveillance-test/cam1.mp4",
        'location': 'Highway Exit 42'
    },
    'cam3': {
        'path': "/kaggle/input/road-surveillance-test/cam3.mp4",
        'location': 'Downtown Plaza'
    }
}

CONFIDENCE_THRESHOLD = 0.75
COLLISION_THRESHOLD = 155
FIGHT_PROXIMITY_THRESHOLD = 80

class Command(BaseCommand):
    help = 'Runs the Surveillance System Engine'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS('🚀 Engine Initializing...'))
        
        # 1. Initialize Global Tools
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8m.pt').to(self.device)
        self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
        self.thread_lock = threading.Lock()
        
        self.stdout.write(f"   - Device: {self.device}")
        self.stdout.write(f"   - Cameras Loaded: {len(CAMERA_FEEDS)}")

        # 2. Main Loop
        try:
            while True:
                # A. Check for Manual User Searches from Dashboard
                self.check_manual_searches()

                # B. Scan the Main Anomaly Feed
                self.process_anomaly_feed()
                
                # Sleep briefly to mimic real-time processing interval
                time.sleep(2) 

        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('🛑 Engine stopping...'))

    # ==========================================
    # 🕵️ MANUAL SEARCH LOGIC
    # ==========================================
    def check_manual_searches(self):
        pending = SearchLog.objects.filter(is_processed=False)
        for search in pending:
            self.stdout.write(self.style.NOTICE(f"🔍 Processing User Search: {search.license_plate}"))
            
            # Create Vehicle Object
            vehicle_obj = Vehicle.objects.create(
                license_plate=search.license_plate, 
                search_log=search
            )
            
            # Trigger Multi-Camera Tracking
            self.run_parallel_tracking(vehicle_obj)
            
            search.is_processed = True
            search.save()

    # ==========================================
    # 🚨 ANOMALY FEED LOGIC
    # ==========================================
    def process_anomaly_feed(self):
        cap = cv2.VideoCapture(INCIDENT_VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_idx = 0
        
        # We only process a chunk of the video to simulate "live" feed
        # In a real loop, you'd manage state to not re-process the whole file
        
        detected_incident = None
        involved_cars = []
        plate_ballot = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_idx += 1
            
            # Optimization: Skip frames
            if frame_idx % 3 != 0: continue

            results = self.model.track(frame, persist=True, verbose=False, conf=0.4)
            incident_type, vehicle_ids = self.detect_incident_logic(results, frame)

            if incident_type:
                # Save Incident to DB
                self.save_incident_to_db(incident_type, frame, vehicle_ids)
                
                if incident_type == 'accident':
                     # Extract Plates
                    detected_plates = self.extract_plates_from_ids(frame, results, vehicle_ids)
                    if detected_plates:
                        self.stdout.write(self.style.SUCCESS(f"   -> Found Plates: {detected_plates}"))
                        # AUTO-TRACKING INITIATED
                        self.initiate_auto_tracking(detected_plates)
                
                # Break to avoid creating 100 incidents for the same event in this loop
                break 
        
        cap.release()

    def detect_incident_logic(self, results, frame):
        """Your logic adapted for the Class"""
        if results[0].boxes.id is None: return None, None
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        
        people = []
        cars = []

        for box, id, cls in zip(boxes, ids, clss):
            center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
            name = results[0].names[int(cls)]
            if name == 'person': people.append({'center': center})
            elif name == 'car': cars.append({'id': id, 'center': center, 'box': box})

        # Fight
        if len(people) >= 2:
            for i in range(len(people)):
                for j in range(i+1, len(people)):
                    dist = np.linalg.norm(np.array(people[i]['center']) - np.array(people[j]['center']))
                    if dist < FIGHT_PROXIMITY_THRESHOLD: return 'fight', None

        # Accident
        if len(cars) >= 2:
             for i in range(len(cars)):
                for j in range(i+1, len(cars)):
                    dist = np.linalg.norm(np.array(cars[i]['center']) - np.array(cars[j]['center']))
                    if dist < COLLISION_THRESHOLD: 
                        return 'accident', [cars[i], cars[j]]
        return None, None

    def save_incident_to_db(self, i_type, frame, vehicle_ids):
        # Prevent duplicate spam (simple timestamp check logic could go here)
        last_incident = Incident.objects.last()
        if last_incident and (time.time() - last_incident.detected_at.timestamp() < 10):
            return # Skip if incident just logged

        _, buffer = cv2.imencode('.jpg', frame)
        file_content = ContentFile(buffer.tobytes(), name=f'{i_type}.jpg')
        
        Incident.objects.create(
            incident_type=i_type,
            location="Main Feed (Anomaly)",
            snapshot=file_content
        )
        self.stdout.write(self.style.WARNING(f"🚨 {i_type.upper()} LOGGED TO DB"))

    def extract_plates_from_ids(self, frame, results, target_car_objs):
        """Extracts plates specifically from cars involved in accident"""
        plates = []
        for car in target_car_objs:
            box = car['box'].astype(int)
            # Crop slightly larger
            crop = frame[max(0, box[1]):box[3], max(0, box[0]):box[2]]
            
            # Forensic Upscale
            processed = self.forensic_upscale(crop)
            if processed is not None:
                txt = self.perform_ocr(processed)
                if txt and len(txt) > 4:
                    plates.append(txt)
        return list(set(plates))

    def initiate_auto_tracking(self, plates):
        """Creates DB Vehicle entries and starts tracking"""
        incident_ref = Incident.objects.last()
        for plate in plates:
            v = Vehicle.objects.create(license_plate=plate, incident=incident_ref)
            self.run_parallel_tracking(v)

    # ==========================================
    # 🎥 MULTI-CAMERA TRACKING
    # ==========================================
    def run_parallel_tracking(self, vehicle_obj):
        target_plate = vehicle_obj.license_plate
        self.stdout.write(f"   ⚡ Launching Tracking Threads for {target_plate}...")

        threads = []
        # Use the HARDCODED CAMERA_FEEDS
        for cam_id, cam_info in CAMERA_FEEDS.items():
            t = threading.Thread(
                target=self.scan_single_camera,
                args=(cam_id, cam_info, vehicle_obj)
            )
            threads.append(t)
            t.start()
        
        for t in threads: t.join()

    def scan_single_camera(self, cam_id, cam_info, vehicle_obj):
        """Worker thread for a specific video file"""
        cap = cv2.VideoCapture(cam_info['path'])
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Load local model instance for thread safety if needed
        # Or use global self.model with lock (YOLO is generally thread-safe for inference but be careful)
        
        frame_idx = 0
        target = vehicle_obj.license_plate

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_idx += 1
            
            # Skip frames for speed
            if frame_idx % 5 != 0: continue

            # Just run detection on the whole frame
            results = self.model.predict(frame, verbose=False, conf=0.4, classes=[2]) # class 2 is car in COCO
            
            for box in results[0].boxes.xyxy:
                x1,y1,x2,y2 = map(int, box[:4])
                car_crop = frame[y1:y2, x1:x2]
                
                # OCR check
                processed = self.forensic_upscale(car_crop)
                if processed:
                    txt = self.perform_ocr(processed)
                    if txt and self.is_similar(target, txt):
                        
                        # MATCH FOUND! Save to DB
                        timestamp_str = self.frame_to_timestamp(frame_idx, fps)
                        
                        _, buffer = cv2.imencode('.jpg', frame)
                        snap = ContentFile(buffer.tobytes(), name=f'{target}_{cam_id}.jpg')

                        VehicleDetection.objects.create(
                            vehicle=vehicle_obj,
                            camera_name=cam_id,                  # Storing string
                            camera_location=cam_info['location'], # Storing string
                            timestamp=timestamp_str,
                            matched_text=txt,
                            snapshot=snap,
                            frame_number=frame_idx
                        )
                        # We found it in this frame, move to next frame logic
                        break 
        cap.release()

    # ==========================================
    # 🛠️ UTILS
    # ==========================================
    def forensic_upscale(self, img):
        if img.size == 0: return None
        try:
            img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(gray)
        except: return None

    def perform_ocr(self, img):
        try:
            res = self.reader.readtext(img, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            txt = "".join(res).upper()
            return "".join(e for e in txt if e.isalnum())
        except: return None

    def is_similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio() > 0.6

    def frame_to_timestamp(self, frame, fps):
        seconds = frame / fps
        return f"{int(seconds//60):02d}:{int(seconds%60):02d}"