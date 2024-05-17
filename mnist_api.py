from fastapi import FastAPI, File, UploadFile, Request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import sys

import time


from prometheus_client import Summary, start_http_server, Counter, Gauge
from prometheus_client import disable_created_metrics

REQUEST_DURATION = Summary('api_timing', 'Request duration in seconds')
counter = Counter('api_call_counter', 'number of times that API is called', ['endpoint', 'client'])
gauge = Gauge('api_runtime_secs', 'runtime of the method in seconds', ['endpoint', 'client'])

app = FastAPI()

def load(path: str):
    return load_model(path)

def predict_digit(model, data_point):
    data_point = np.array(data_point).reshape(1, 28* 28)
    prediction = model.predict(data_point)
    return str(np.argmax(prediction))

def format_image(contents):
    img = Image.open(BytesIO(contents)).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img)
    data_point = img_array.flatten() / 255.0  # Flatten and normalize
    return data_point.tolist()

@REQUEST_DURATION.time()
@app.post('/predict')
async def predict(request:Request, file: UploadFile = File(...)):
    counter.labels(endpoint='/predict', client=request.client.host).inc()
    
    start = time.time()
    
    contents = await file.read()
    data_point = format_image(contents)
    model_path = sys.argv[1]  # Get model path from command line argument
    model = load(model_path)
    digit = predict_digit(model, data_point)
    
    time_taken = time.time() - start
    
    gauge.labels(endpoint='/predict', client=request.client.host).set(time_taken)
    
    return {"digit": digit}

if __name__ == "__main__":
    import uvicorn
    model_path = sys.argv[1]  # Get model path from command line argument
    model = load(model_path)
    start_http_server(10000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
