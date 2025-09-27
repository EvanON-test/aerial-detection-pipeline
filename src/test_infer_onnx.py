import cv2, numpy as np, onnxruntime as ort, time

# Use Pi Camera (libcamera). Change to v4l2src if using USB webcam
GST_PIPE = "libcamerasrc ! video/x-raw,width=416,height=416,framerate=30/1 ! videoconvert ! appsink"
cap = cv2.VideoCapture(GST_PIPE, cv2.CAP_GSTREAMER)

# Load ONNX model
sess = ort.InferenceSession("models/yolov8n-sim.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
imgsz = 416

def preprocess(bgr):
    resized = cv2.resize(bgr, (imgsz, imgsz))
    x = resized[:, :, ::-1].transpose(2,0,1)[None].astype(np.float32) / 255.0
    return x

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera read failed")
        break
    x = preprocess(frame)
    start = time.time()
    _ = sess.run(None, {input_name: x})[0]
    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Pi-ONNX", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

