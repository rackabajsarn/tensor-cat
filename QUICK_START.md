# Quick Start Guide - LocalAI Branch

## Setup Checklist

### Prerequisites
- [ ] ESP32-WROVER with PSRAM and SD card
- [ ] Raspberry Pi with Coral TPU USB accelerator
- [ ] MQTT broker running
- [ ] Home Assistant configured

---

## Initial Setup

### 1. Configure Credentials

**tensor-cat/credentials.py**
```python
WIFI_SSID = "your_wifi"
WIFI_PASSWORD = "your_password"
MQTT_SERVER = "192.168.1.x"
SECRET_KEY = "your_secret_key"
ESP32_IP = "192.168.1.14"  # ← ADD THIS
```

**CatCam2/src/secrets.h**
```cpp
#define WIFI_SSID "your_wifi"
#define WIFI_PASSWORD "your_password"
#define MQTT_SERVER "192.168.1.x"
```

### 2. Install Dependencies

**ESP32 (PlatformIO)**
```bash
cd CatCam2
pio lib install  # Installs: PubSubClient, ArduinoJson, ArduTFLite, ESP Async WebServer
```

**Server (Python)**
```bash
cd tensor-cat
pip install -r requirements.txt
```

### 3. Compile and Upload ESP32
```bash
cd CatCam2
pio run --target upload
# Monitor serial output
pio device monitor
```

### 4. Start Server

**On Server (Raspberry Pi):**
```bash
cd tensor-cat
python app.py
# Access web UI at http://SERVER_IP:5000
```

**For Local Development (Windows/Mac without Coral TPU):**
```bash
cd tensor-cat
# Set offline mode to disable MQTT and Coral TPU features
set LOCAL_OFFLINE_MODE=1   # Windows CMD
# or
$env:LOCAL_OFFLINE_MODE="1"   # PowerShell
# or
export LOCAL_OFFLINE_MODE=1   # Linux/Mac

python app.py
```

> **Note:** Offline mode disables MQTT listener, Coral TPU inference, and ESP32 uploads. Training, labeling, and model versioning work normally.

---

## Training Workflow

### Option A: Via Web UI
1. Navigate to http://SERVER_IP:5000/model
2. Adjust parameters if needed:
   - Epochs: 10 (Coral model)
   - Simple model epochs: 40 (ESP32 model)
3. Click "Retrain Model"
4. Monitor progress bar (0-50%: Coral, 50-100%: ESP32)
5. Wait for automatic upload to ESP32

### Option B: Command Line
```bash
cd tensor-cat

# Train both models
python train_model.py --epochs 10 --fine_tune_epochs 5
python train_simple_model.py --epochs 40

# Upload to ESP32 (replace <RUN_ID> with the folder under models/local created by the training run)
curl -F "file=@models/local/<RUN_ID>/model/my_simple_model_quant.tflite" http://192.168.1.14/upload
```

---

## Testing

### 1. Check ESP32 Status
```bash
curl http://192.168.1.14/status
```

Expected output:
```
OK
Model: SD Card
Free Heap: 150000
Uptime: 123s
```

### 2. Test Model Upload
```bash
# From tensor-cat directory
curl -F "file=@models/local/<RUN_ID>/model/my_simple_model_quant.tflite" http://192.168.1.14/upload
```

Expected: `Model uploaded successfully`

### 3. Monitor MQTT Topics
```bash
# Subscribe to all catflap topics
mosquitto_sub -h MQTT_BROKER -t "catflap/#" -v
```

Watch for:
- `catflap/esp32_inference` - Local inference results
- `catflap/server_inference` - Server inference results
- `catflap/inference_comparison` - Accuracy metrics
- `catflap/image` - Images sent to server

### 4. Trigger Capture
```bash
mosquitto_pub -h MQTT_BROKER -t "catflap/command" -m "snapshot"
```

---

## Monitoring in Home Assistant

### Add to configuration.yaml
```yaml
mqtt:
  sensor:
    - name: "Cat Flap ESP32 Inference"
      state_topic: "catflap/esp32_inference"
      icon: mdi:chip
      
    - name: "Cat Flap Server Inference"
      state_topic: "catflap/server_inference"
      icon: mdi:cloud
      
    - name: "Model Agreement"
      state_topic: "catflap/inference_comparison"
      icon: mdi:chart-line
      
    - name: "Cat Flap Roundtrip Time"
      state_topic: "catflap/roundtrip"
      unit_of_measurement: "ms"
      icon: mdi:timer
```

### Create Dashboard Card
```yaml
type: entities
title: Cat Flap AI
entities:
  - entity: sensor.cat_flap_esp32_inference
    name: ESP32 (Local)
  - entity: sensor.cat_flap_server_inference
    name: Server (Coral)
  - entity: sensor.cat_flap_model_agreement
    name: Agreement Rate
  - entity: sensor.cat_flap_roundtrip_time
    name: Response Time
  - entity: switch.cat_flap_enable
  - entity: camera.cat_flap_last_image
```

---

## Common Tasks

### Switch Model Source (ESP32)
```bash
# Use SD card model
mosquitto_pub -h MQTT_BROKER -t "catflap/model_source/set" -m "ON"

# Use embedded model
mosquitto_pub -h MQTT_BROKER -t "catflap/model_source/set" -m "OFF"
```

### Force Snapshot
```bash
mosquitto_pub -h MQTT_BROKER -t "catflap/command" -m "snapshot"
```

### Manual Flap Control
```bash
# Open flap
mosquitto_pub -h MQTT_BROKER -t "catflap/flap_state/set" -m "ON"

# Close flap
mosquitto_pub -h MQTT_BROKER -t "catflap/flap_state/set" -m "OFF"
```

### Check Model Status
```bash
# Via HTTP
curl http://ESP32_IP/status

# Via MQTT
mosquitto_sub -h MQTT_BROKER -t "catflap/debug" -v
```

---

## Troubleshooting

### ESP32 Not Connecting to WiFi
1. Check credentials in `secrets.h`
2. Verify WiFi signal strength
3. Check serial monitor: `pio device monitor`

### Model Upload Fails
1. Check ESP32 IP: `ping 192.168.1.14`
2. Verify SD card inserted
3. Check file size: `ls -lh models/local/<RUN_ID>/model/*.tflite`
4. Review ESP32 serial output

### Inference Not Working
1. Check model loaded: `curl http://ESP32_IP/status`
2. Verify MQTT broker running
3. Test with snapshot command
4. Check tensor arena size (20KB default)

### Models Disagree Frequently
1. Check `catflap/inference_comparison` for accuracy
2. Review mismatch patterns in logs
3. Collect more training data
4. Consider adjusting class weights

### Server Not Receiving Images
1. Check MQTT topic subscription
2. Verify image published: `mosquitto_sub -h MQTT_BROKER -t "catflap/image"`
3. Check network bandwidth
4. Review app.py logs

---

## Performance Tuning

### If ESP32 Inference Too Slow
- Reduce tensor arena size
- Simplify model architecture
- Optimize preprocessing

### If Server Inference Too Slow
- Check Coral TPU connection
- Verify Edge TPU compiler used
- Monitor CPU/memory usage

### If Memory Issues
- Reduce image sizes
- Adjust PSRAM allocation
- Use smaller model

---

## Data Collection Tips

### For Better Training
1. Capture varied lighting conditions
2. Include different angles
3. Balance classes (equal samples)
4. Label consistently
5. Remove ambiguous images

### Via Web UI
1. Navigate to http://SERVER_IP:5000/classify
2. Review unlabeled images
3. Toggle labels (cat, morris, entering, prey)
4. Click save to add to dataset
5. Retrain when you have 100+ new images

---

## Backup & Recovery

### Backup Current Model
```bash
# ESP32 embedded model
cp CatCam2/lib/model/model.cc model_backup_$(date +%Y%m%d).cc

# Server models
cp model/my_model_quant_edgetpu.tflite model_backup_$(date +%Y%m%d).tflite
cp models/local/<RUN_ID>/model/my_simple_model_quant.tflite local_simple_backup_$(date +%Y%m%d).tflite
```

### Restore Model
```bash
# Upload to ESP32 SD card
curl -F "file=@local_simple_backup_20250101.tflite" http://ESP32_IP/upload

# Or replace embedded model
cp model_backup_20250101.cc CatCam2/lib/model/model.cc
pio run --target upload
```

---

## Next Steps

1. ✅ Complete initial setup
2. ✅ Test with manual snapshot
3. ✅ Collect 100+ labeled images
4. ✅ Train first models
5. ✅ Monitor accuracy for 1 week
6. ✅ Retrain with new data
7. ✅ Fine-tune based on performance

---

## Support

- Check `INTEGRATION_SUMMARY.md` for detailed architecture
- Review ESP32 serial output for debugging
- Check `tensor-cat/logs/app.log` for server issues
- Monitor Home Assistant MQTT logs

## Key Files

**ESP32**
- `src/main.cpp` - Main firmware
- `lib/model/model.h` & `model.cc` - Embedded model
- `platformio.ini` - Build configuration

**Server**
- `app.py` - Flask application
- `train_model.py` - Coral TPU training
- `train_simple_model.py` - ESP32 model training
- `credentials.py` - Configuration

---

Good luck with your AI cat flap! 🐱🚪
