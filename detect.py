# detect.py
import cv2
import numpy as np
import time
from tensorflow.lite.python.interpreter import Interpreter

# --- Load model ---
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Usually there is a single input tensor; handle multiple just in case
inp = input_details[0]
input_shape = inp['shape']  # e.g. [1, 224, 224, 3]
_, height, width, channels = input_shape
input_dtype = inp['dtype']

print(f"Model input shape: {input_shape}, dtype: {input_dtype}")

# --- Load labels ---
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

print("Labels:", labels)

# --- Helper: preprocess frame for model ---
def preprocess(frame):
    # frame: BGR image from OpenCV
    # 1. convert BGR -> RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 2. resize to model size
    img = cv2.resize(img, (width, height))
    # 3. normalize depending on dtype
    if input_dtype == np.float32:
        img = img.astype(np.float32) / 255.0  # scale to [0,1]
    else:
        img = img.astype(input_dtype)
    # 4. add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# --- Start webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Make sure webcam is available.")

# Small smoothing for display
last_label = ""
last_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        inp_img = preprocess(frame)

        # set input tensor and run
        interpreter.set_tensor(input_details[0]['index'], inp_img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # shape depends on model

        # If output is a vector of probabilities:
        if output_data.ndim == 1:
            idx = int(np.argmax(output_data))
            confidence = float(output_data[idx])
        else:
            # fallback: flatten and take max
            flat = output_data.flatten()
            idx = int(np.argmax(flat))
            confidence = float(flat[idx])

        label = labels[idx] if idx < len(labels) else f"Label {idx}"
        now = time.time()
        # display label every frame but throttle prints to console
        if now - last_time > 0.5:
            print(f"Detected: {label} ({confidence*100:.1f}%)")
            last_time = now

        # Draw on frame
        color = (0, 255, 0) if label.lower().startswith("plastic") else (0, 0, 255)
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("Plastic Detector (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
