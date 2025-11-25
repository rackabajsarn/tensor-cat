# CatCam2 + tensor-cat Integration - LocalAI Branch

## Overview
This integration enables **dual-stage inference** for the cat flap system:
1. **ESP32 (Edge)**: Fast local inference on 96x96 grayscale images for immediate prey/no_prey detection
2. **Coral TPU (Server)**: Detailed inference on 384x384 images for full classification (5 classes)

The system automatically trains both models and compares their performance for continuous improvement.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          tensor-cat Server                        │
│                        (Linux + Coral TPU)                        │
│                                                                    │
│  1. Trains MobileNetV2 (224x224, 5 classes)                      │
│  2. Trains Simple CNN (96x96, 3 classes)                         │
│  3. Uploads simple model to ESP32 via HTTP                       │
│  4. Receives images from ESP32 for detailed inference             │
│  5. Publishes comparison metrics                                  │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        │ MQTT + HTTP
                        │
┌───────────────────────┴──────────────────────────────────────────┐
│                         CatCam2 ESP32                             │
│                  (ESP32-WROVER + OV2640 Camera)                   │
│                                                                    │
│  1. Captures 640x480 image                                        │
│  2. Crops to 384x384                                              │
│  3. Resizes to 96x96 for local inference                         │
│  4. Runs TFLite Micro inference (prey/not_prey)                  │
│  5. Makes immediate flap open/close decision                      │
│  6. Sends 384x384 image to server for detailed inference         │
│  7. Compares results and publishes metrics                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## New MQTT Topics

### ESP32 → Home Assistant
- `catflap/esp32_inference` - Local ESP32 inference result (prey/not_prey)
- `catflap/server_inference` - Server inference result (prey/not_prey)
- `catflap/inference_comparison` - Accuracy comparison (e.g., "Matches: 45/50 (90.0%)")

### Existing Topics (Enhanced)
- `catflap/inference` - Full server classification (5 classes)
- `catflap/image` - 384x384 cropped image for server inference

---

## Model Training Workflow

### On Server (tensor-cat)
When user clicks "Retrain Model" in web UI:

1. **Train Coral TPU Model** (Progress: 0-50%)
   - Script: `train_model.py`
   - Input: 224x224 RGB images
   - Classes: not_cat, unknown_cat_entering, cat_morris_leaving, cat_morris_entering, prey
   - Output: `model/my_model_quant_edgetpu.tflite`

2. **Train ESP32 Simple Model** (Progress: 50-100%)
   - Script: `train_simple_model.py`
   - Input: 96x96 grayscale images
   - Classes: not_cat, not_prey, prey
   - Output: `simple_model/my_simple_model_quant.tflite`

3. **Upload to ESP32**
   - HTTP POST to `http://ESP32_IP/upload`
   - File: `my_simple_model_quant.tflite`
   - ESP32 validates and saves to SD card

---

## ESP32 Model Upload Server

### Endpoint: `/upload` (POST)
- Receives TFLite model file
- Validates:
  - Size: 1KB - 2MB
  - Format: TFLite magic bytes (`TFL3` at offset 4-7)
- Saves to: `/model.tflite` on SD card
- Atomically replaces old model (temp file → rename)
- Optionally reloads model if using SD card source

### Endpoint: `/status` (GET)
- Returns system status
- Model source (Embedded/SD Card)
- Free heap
- Uptime

---

## Inference Pipeline

### ESP32 Local Inference
```cpp
1. Capture 640x480 grayscale image
2. Crop to 384x384 (center crop)
3. Resize to 96x96 for inference
4. Run TFLite Micro inference
   - Input: 96x96 uint8 grayscale
   - Output: 3 classes (not_cat, not_prey, prey)
5. Decision:
   - prey → Close flap, publish "prey"
   - not_prey → Open flap, publish "not_prey"
6. Convert 384x384 to JPEG
7. Send to server via MQTT
```

### Server Detailed Inference
```python
1. Receive 384x384 JPEG from MQTT
2. Preprocess to 224x224 RGB
3. Run Coral TPU inference
   - Output: 5 classes (full classification)
4. Simplify result to prey/not_prey
5. Publish both detailed and simplified results
6. Save image with EXIF labels
```

### Comparison Tracking
```cpp
1. ESP32 subscribes to server_inference
2. When server result arrives:
   - Compare: lastESP32Inference vs lastServerInference
   - Increment: inferenceMatches or inferenceMismatches
   - Calculate accuracy
   - Publish comparison stats
```

---

## Configuration

### tensor-cat `credentials.py`
Add ESP32 IP address:
```python
ESP32_IP = '192.168.1.14'  # Your ESP32 IP
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_OFFLINE_MODE` | `0` | Set to `1` for local development without Coral TPU/MQTT |

**Offline Mode** disables:
- MQTT listener and publishing
- Coral TPU inference (returns dummy class)
- ESP32 model upload

**Offline Mode keeps working:**
- Web UI for labeling and training
- Model versioning (save/load/activate/delete)
- Local model training
- All report viewing

**Usage:**
```bash
# Windows CMD
set LOCAL_OFFLINE_MODE=1
python app.py

# PowerShell
$env:LOCAL_OFFLINE_MODE="1"
python app.py

# Linux/Mac
export LOCAL_OFFLINE_MODE=1
python app.py
```

### ESP32 `platformio.ini`
Dependencies added:
- `ESP Async WebServer` - For HTTP model upload

### ESP32 SD Card Structure
```
/
├── model.tflite          # Active simple model
└── model_temp.tflite     # Temporary upload file
```

---

## Deployment Steps

### 1. Upload ESP32 Firmware
```bash
cd CatCam2
pio run --target upload
```

### 2. Train Both Models
```bash
cd tensor-cat
# Via web UI: http://SERVER_IP:5000/model
# Click "Retrain Model"
# Or manually:
python train_model.py --epochs 10 --fine_tune_epochs 5
python train_simple_model.py --epochs 40
```

### 3. Upload Simple Model to ESP32
Automatic after training completes, or manually:
```bash
curl -F "file=@simple_model/my_simple_model_quant.tflite" http://ESP32_IP/upload
```

### 4. Monitor in Home Assistant
Add sensors for:
- `sensor.cat_flap_esp32_inference`
- `sensor.cat_flap_server_inference`
- `sensor.cat_flap_inference_comparison`

---

## Flask Service (Server)

The tensor-cat Flask app runs as a systemd service on the server.

### Service Configuration

**Service file location:** `/etc/systemd/system/flaskapp.service`

**Virtual environment:**
- Python: `/venv/coral/bin/python`
- Activate: `source /venv/coral/bin/activate`

### Updating the Flask Application

1. **Make your changes** to `app.py`, templates, or other files

2. **Restart the service:**
   ```bash
   sudo systemctl restart flaskapp
   ```
   If you modified the service file itself:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart flaskapp
   ```

3. **Check service status:**
   ```bash
   sudo systemctl status flaskapp
   ```

4. **Clear browser cache** (for static file changes):
   - Hard refresh: `Ctrl+F5`
   - Or clear cache manually

### Testing Changes Locally

Before restarting the service, test changes manually:
```bash
source /venv/coral/bin/activate
python /home/app/app.py
```

### Monitoring Logs

View service logs for debugging:
```bash
journalctl -xe -u flaskapp
```

Follow logs in real-time:
```bash
journalctl -f -u flaskapp
```

### Tips

- **Minimize downtime**: Schedule updates during low-activity periods
- **Test first**: Run manually before restarting the service
- **Monitor logs**: Watch for errors after updates
- **Static files**: Browser may cache CSS/JS - use hard refresh

---

## Model Comparison Example

```yaml
# Home Assistant configuration.yaml
sensor:
  - platform: mqtt
    name: "ESP32 Inference"
    state_topic: "catflap/esp32_inference"
    
  - platform: mqtt
    name: "Server Inference"
    state_topic: "catflap/server_inference"
    
  - platform: mqtt
    name: "Model Accuracy"
    state_topic: "catflap/inference_comparison"
```

---

## Benefits

### Edge Inference (ESP32)
- ⚡ **Fast**: ~50-100ms response time
- 🔒 **Reliable**: Works without server/network
- 🎯 **Focused**: Simple prey/not_prey decision
- 💾 **Memory Efficient**: 20KB tensor arena

### Server Inference (Coral TPU)
- 🎯 **Accurate**: Full 5-class classification
- 🧠 **Detailed**: Distinguishes Morris, unknown cats, direction
- 📊 **Trackable**: Stores images with labels for retraining
- 🔍 **Verifiable**: Comparison with edge model

### Continuous Improvement
- 📈 Track edge vs server accuracy
- 🔄 Retrain with misclassified images
- ⚖️ Balance speed vs accuracy
- 🎓 Learn from real-world data

---

## Troubleshooting

### Model Upload Fails
- Check ESP32 IP address in `credentials.py`
- Verify SD card is formatted and inserted
- Check file size (should be 50-200KB for simple model)
- View ESP32 serial output for errors

### Inference Mismatch
- Normal for different model architectures
- Track mismatch patterns in logs
- Use mismatches to identify training data gaps
- Consider increasing simple model complexity if needed

### SD Card Model Not Loading
- Check `/status` endpoint: `http://ESP32_IP/status`
- Verify model file exists on SD card
- Check TFLite magic bytes validation
- Fall back to embedded model automatically

---

## Next Steps

1. **Monitor Performance**
   - Track inference comparison over time
   - Identify common mismatch scenarios
   - Adjust class weights if needed

2. **Optimize Simple Model**
   - Experiment with model architecture
   - Balance size vs accuracy
   - Consider quantization aware training

3. **Enhance Comparison**
   - Add confusion matrix for edge model
   - Track per-class accuracy
   - Log images where models disagree

4. **Automate Retraining**
   - Schedule periodic retraining
   - Use mismatch images for targeted training
   - Auto-deploy models when accuracy improves

---

## File Changes Summary

### tensor-cat
- ✅ `app.py` - Added dual model training and upload
- ✅ `train_simple_model.py` - Simple CNN for ESP32
- ✅ `credentials.py` - Add ESP32_IP configuration

### CatCam2
- ✅ `platformio.ini` - Added ESP Async WebServer
- ✅ `src/main.cpp` - Added HTTP server, dual inference, comparison
- ✅ `lib/model/model.h` - Const model declaration
- ✅ `lib/model/model.cc` - Const model data

---

## Performance Metrics

### Typical Values
- **ESP32 Inference**: 50-100ms
- **Total Roundtrip**: 200-500ms (with server inference)
- **Model Size**: 
  - Simple (ESP32): 50-150KB
  - MobileNetV2 (Server): 2-4MB
- **Memory Usage**:
  - ESP32 Tensor Arena: 20KB
  - ESP32 Model: Stored in PSRAM/SD

### Target Accuracy
- **Edge Model**: >85% on prey detection
- **Server Model**: >90% on full classification
- **Agreement**: >80% between edge and server

---

## Credits
Implementation for LocalAI branch integrating:
- TensorFlow Lite Micro on ESP32
- Google Coral TPU on Linux server
- Dual-stage inference with comparison
- Automated model training and deployment
