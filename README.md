<<<<<<< HEAD
# 🛒 AI-Powered Shoplifting Detection System

 

## 📌 Problem Statement

Retail theft, particularly shoplifting, costs the global retail industry billions of dollars annually. Most small and medium-sized retailers lack access to real-time, intelligent surveillance systems that can proactively detect and alert on suspicious activities. Current CCTV systems are passive, relying on human monitoring or after-the-fact footage analysis, which is both time-consuming and ineffective.

## 💡 Our Approach & Solution

We propose an **AI-powered Shoplifting Detection System** that utilizes deep learning and computer vision to analyze CCTV footage in real-time and detect suspicious behaviors linked to shoplifting. This system can be integrated with existing surveillance infrastructure and provides instant alerts, thereby preventing loss and enhancing store security.

### Key Steps in Our Solution:
1. **Data Collection** – Simulated surveillance footage of shoplifting and normal activity.
2. **Preprocessing** – Extract video frames, resize, normalize, and prepare for model input.
3. **Model Training** – Train a custom CNN/LSTM hybrid model to classify sequences as "Shoplifting" or "Normal Activity".
4. **Flask API** – Serve predictions through a lightweight REST API.
5. **Frontend  ** – GUI for Monitoring and receiving results.

## 🚀 Features

- 🎯 **Real-Time Detection:** Detects shoplifting activity from CCTV video feeds.
- 🔔 **Instant Alerts:** Returns prediction with confidence score for immediate action.
- 🧠 **Deep Learning Powered:** Uses a pre-trained TensorFlow model.
- 📦 **Easy Integration:** REST API built with Flask, deployable on-premises or in the cloud.
- 🔐 **Secure Access (optional):** Can be extended with API key-based access.
- 💻 **Cross-Platform:** Compatible with web, desktop, or mobile frontends.

## 🚀 Features

| Feature             | Description                                         |
|---------------------|-----------------------------------------------------|
| 🎯 Real-Time AI      | Predicts shoplifting from live CCTV feeds           |
| 🔔 Instant Alerts    | On-screen visual alerts for suspicious activity     |
| 🧠 Deep Learning     | Custom-trained TensorFlow model                     |
| 📦 Easy Integration  | Flask-based API, deployable anywhere                |
| 🔐 Secure (Optional) | Extend with login/API key auth                      |
| 💻 Responsive UI     | Built with HTML, CSS (Tailwind-ready)               |

---


## 🛠️ Tech Stack

| Component      | Technology         |
|----------------|--------------------|
| AI Framework   | TensorFlow, Keras  |
| Video Handling | OpenCV             |
| Backend API    | Flask (Python)     |
| Deployment     | Localhost , Docker |
| Programming    | Python 3.x         |
| Frontend       |  HTML/CSS |
| Testing        | Postman / cURL     |



## 🧠 Model Architecture & Training

The detection model (`model.h5`) is a custom-trained **YOLO + CNN + LSTM hybrid**, built using real CCTV surveillance footage captured from retail environments.

### 🔬 Architecture Highlights:

| Component | Purpose |
|----------|---------|
| **YOLO (You Only Look Once)** | Real-time person/object detection in video frames |
| **CNN (Convolutional Neural Network)** | Spatial feature extraction from detected ROIs |
| **LSTM (Long Short-Term Memory)** | Temporal modeling of suspicious behavior sequences |

---

### 📦 Training Details

- **Input Data**: Real CCTV surveillance video clips (Suspicious & normal behavior)
- **Frame Sampling**: Key frames extracted at consistent intervals
- **Preprocessing**: Resized to 224×224, normalized, and annotated
- **Labels**: Binary (0 = Normal, 1 = Suspicious)
- **Training Time**: ~12 hours on NVIDIA RTX 3060
- **Frameworks**: TensorFlow, Keras, OpenCV, YOLOv5

---

### 🔔 Alert Trigger
- Detection pipeline continuously analyzes frames
- When label = `Shoplifting Detected`, alert is triggered

### 📤 Email Alert Format
- **Subject:** 🔴 Suspicious Activity Detected
- **Body:** Timestamp, frame location, and detection confidence
- **Attachment:** Optional snapshot image (JPEG) of the frame

### 📬 Tech Used
- `smtplib` for SMTP protocol
- `email.mime.multipart` for constructing MIME email
- `email.mime.image` for attaching frame as image
- Works with Gmail, Outlook, or custom SMTP servers

---


## 🌐 Web Architecture (Flask + Frontend)
💻 Full-Stack UI — Flask backend + modern frontend (HTML, CSS, JS)


This project is built as a full-stack AI system using **Flask** for backend routing and **HTML/CSS/JS** (Tailwind CSS) for the frontend UI.

### 🏗️ Flask Backend

- Acts as the bridge between AI inference and web display
- Routes:
  - `/` → Live monitoring page (`stream.html`)
  - `/video` → MJPEG video stream via OpenCV frames
  - `/predict` → Optional POST route for image prediction (extendable)

### 🧩 Flask Responsibilities

| Task                         | Role                                 |
|------------------------------|--------------------------------------|
| Model Loading (`model.h5`)   | Loads once at startup                |
| Frame Processing             | Captures and preprocesses webcam frames |
| Prediction                   | Feeds frames through CNN+LSTM model |
| Video Streaming              | Streams output via MJPEG (Response) |
| Alert Trigger                | Sends email using `smtplib` + `MIME` |

---

### 💻 Frontend UI

Built with modern HTML/CSS (and optionally Tailwind CSS), the UI includes:

| Page         | File             | Description                         |
|--------------|------------------|-------------------------------------|
| Live Monitor | `stream.html`    | Shows real-time video + overlay     |
| Dashboard    | `index.html`     | (Optional) Can show logs or stats   |
| Styling      | `style.css`      | Custom CSS or Tailwind CDN used     |

### 💡 UI Highlights

- Responsive layout with clean design
- Live stream auto-refreshes via MJPEG `<img src="">`
- Clear labels: `Safe` or `Shoplifting Detected`
- Red border or alerts on detection
- Optional footer with branding, timestamp

---

## 📸 Screenshots

 
### 🧠 2\1. Model Output Sample
 ![image](https://github.com/user-attachments/assets/e464ba6c-9322-475c-9492-8bdae78137fa)
 ![image](https://github.com/user-attachments/assets/db5c8b2f-d7fd-4633-84f2-3e6716320bcd)
 
 ##Realtime Alerts
 
 ![image](https://github.com/user-attachments/assets/569923db-5a31-4e9a-948e-ad4ec710c86b)
 ![image](https://github.com/user-attachments/assets/3fdb8305-3595-4a7d-bc45-efe796560ba5)





 
>>>>>>> 0e1216f (Initial commit)
