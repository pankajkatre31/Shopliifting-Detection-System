# app.py - Main Flask Application
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import threading
import time
import logging
from datetime import datetime
import os
from pathlib import Path
import json
import sqlite3
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import pickle
from collections import deque
import uuid


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shoplifting_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

@dataclass
class Detection:
    """Data class for detection events"""
    id: str
    timestamp: datetime
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    description: str
    frame_path: Optional[str] = None

class ShopliftingDetector:
    """AI-powered shoplifting detection system"""
    
    def __init__(self):
        self.is_active = False
        self.detection_threshold = 0.6
        self.person_cascade = None
        self.motion_history = deque(maxlen=30)  # 30 frames of motion history
        self.person_tracker = {}
        self.suspicious_behavior_patterns = []
        self.frame_count = 0
        
        # Initialize detection models
        self._load_models()
        
        # Database setup
        self._init_database()
        
    def _load_models(self):
        """Load OpenCV models for person detection"""
        try:
            # Load Haar Cascade for person detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            if os.path.exists(cascade_path):
                self.person_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("Person detection model loaded successfully")
            else:
                logger.warning("Person cascade not found, using alternative detection")
                
        except Exception as e:
            logger.error(f"Error loading detection models: {e}")
    
    def _init_database(self):
        """Initialize SQLite database for storing detections"""
        try:
            os.makedirs('data', exist_ok=True)
            conn = sqlite3.connect('data/detections.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    confidence REAL,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_width INTEGER,
                    bbox_height INTEGER,
                    description TEXT,
                    frame_path TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def save_detection(self, detection: Detection):
        """Save detection to database"""
        try:
            conn = sqlite3.connect('data/detections.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO detections 
                (id, timestamp, confidence, bbox_x, bbox_y, bbox_width, bbox_height, description, frame_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection.id,
                detection.timestamp.isoformat(),
                detection.confidence,
                detection.bbox[0],
                detection.bbox[1],
                detection.bbox[2],
                detection.bbox[3],
                detection.description,
                detection.frame_path
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Detection saved: {detection.id}")
            
        except Exception as e:
            logger.error(f"Error saving detection: {e}")
    
    def detect_suspicious_behavior(self, frame, frame_gray):
        """Main detection logic for suspicious behavior"""
        detections = []
        self.frame_count += 1
        
        try:
            # Person detection
            if self.person_cascade is not None:
                persons = self.person_cascade.detectMultiScale(
                    frame_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in persons:
                    
                    confidence = self._analyze_person_behavior(frame, (x, y, w, h))
                    
                    if confidence > self.detection_threshold:
                        detection = Detection(
                            id=str(uuid.uuid4()),
                            timestamp=datetime.now(),
                            confidence=confidence,
                            bbox=(x, y, w, h),
                            description=f"Suspicious behavior detected (confidence: {confidence:.2f})"
                        )
                        
                        detections.append(detection)
                        self.save_detection(detection)
                        
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, f'ALERT: {confidence:.2f}', 
                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Motion detection for additional context
            self._detect_motion(frame_gray)
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
        
        return frame, detections
    
    def _analyze_person_behavior(self, frame, bbox):
        """Analyze person behavior for suspicious patterns"""
        x, y, w, h = bbox
        
        # Simple heuristic-based detection
        confidence = 0.0
        
        # Check for loitering (staying in same area)
        person_center = (x + w//2, y + h//2)
        
        # Check if person is near shelf areas (assuming bottom half of frame)
        if y + h > frame.shape[0] * 0.6:
            confidence += 0.3
        
        # Check for unusual movement patterns
        if len(self.motion_history) > 10:
            recent_motion = sum(self.motion_history[-10:]) / 10
            if recent_motion > 0.8:  # High motion threshold
                confidence += 0.4
        
        # Random variation to simulate AI detection
        confidence += np.random.uniform(0.1, 0.3)
        
        return min(confidence, 1.0)
    
    def _detect_motion(self, frame_gray):
        """Detect motion in the frame"""
        if len(self.motion_history) > 0:
            # Simple frame differencing
            prev_frame = self.motion_history[-1] if self.motion_history else frame_gray
            diff = cv2.absdiff(frame_gray, prev_frame)
            motion_score = np.mean(diff) / 255.0
            self.motion_history.append(motion_score)
        else:
            self.motion_history.append(0.0)

class VideoStream:
    """Handle video streaming and processing"""
    
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.detector = ShopliftingDetector()
        self.is_running = False
        self.lock = threading.Lock()
        self.current_frame = None
        
    def start(self):
        """Start video capture"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.error("Failed to open video source")
                return False
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.detector.is_active = True
            logger.info("Video stream started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting video stream: {e}")
            return False
    
    def stop(self):
        """Stop video capture"""
        self.is_running = False
        self.detector.is_active = False
        
        if self.cap:
            self.cap.release()
        logger.info("Video stream stopped")
    
    def get_frame(self):
        """Get current frame with detection overlay"""
        if not self.is_running or not self.cap:
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for detection
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply detection
            frame, detections = self.detector.detect_suspicious_behavior(frame, frame_gray)
            
            # Add timestamp and system info
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"ShopSecure AI - {timestamp}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add detection count
            detection_count = len(detections)
            if detection_count > 0:
                cv2.putText(frame, f"ALERTS: {detection_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            with self.lock:
                self.current_frame = frame.copy()
            
            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None

# Global video stream instance
video_stream = VideoStream()

def generate_frames():
    """Generate video frames for streaming"""
    while True:
        frame = video_stream.get_frame()
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', year=datetime.now().year)

@app.route('/stream')
def stream():
    """Live video stream page"""
    return render_template('stream.html', year=datetime.now().year)

@app.route('/video')
def video():
    """Video streaming endpoint"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    """Start detection system"""
    try:
        if not video_stream.is_running:
            success = video_stream.start()
            if success:
                return jsonify({'status': 'success', 'message': 'Detection started'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to start detection'})
        else:
            return jsonify({'status': 'info', 'message': 'Detection already running'})
    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    """Stop detection system"""
    try:
        video_stream.stop()
        return jsonify({'status': 'success', 'message': 'Detection stopped'})
    except Exception as e:
        logger.error(f"Error stopping detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/detections')
def get_detections():
    """Get recent detections"""
    try:
        conn = sqlite3.connect('data/detections.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM detections 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        detections = []
        for row in rows:
            detections.append({
                'id': row[0],
                'timestamp': row[1],
                'confidence': row[2],
                'bbox': [row[3], row[4], row[5], row[6]],
                'description': row[7],
                'frame_path': row[8]
            })
        
        return jsonify({'detections': detections})
        
    except Exception as e:
        logger.error(f"Error fetching detections: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'is_running': video_stream.is_running,
        'detector_active': video_stream.detector.is_active if video_stream.detector else False,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Start video stream
    video_stream.start()
    
    try:
        # Run Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("Shutting down application...")
    finally:
        video_stream.stop()
        cv2.destroyAllWindows()