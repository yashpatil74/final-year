# **Wildfire Detection System Using CNNs**

## **Overview**
The Wildfire Detection System project aims to leverage advanced Convolutional Neural Networks (CNNs) for accurate and real-time detection of wildfires. This system is designed to operate efficiently in resource-constrained environments, making it ideal for deployment on edge devices such as Raspberry Pi. By integrating machine learning, edge computing, and real-time monitoring, the system provides a scalable and actionable solution for wildfire management.

Key Features:
- Benchmarking CNN architectures (ConvNeXt-Tiny, EfficientNetB0, MobileNetV2, ResNet18).
- Real-time image classification into wildfire and non-wildfire categories.
- Deployment on edge devices for low-latency processing.
- Enhanced robustness through data augmentation and hyperparameter tuning.

---

## **Table of Contents**
1. [Motivation](#motivation)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Setup Instructions](#setup-instructions)
5. [Dataset](#dataset)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Results](#results)
8. [Edge Deployment](#edge-deployment)
9. [Future Scope](#future-scope)

---

## **Motivation**
Wildfires cause catastrophic damage to ecosystems, economies, and human lives. Current detection methods often face limitations like delayed response times and inefficiencies in remote or inaccessible regions. This project aims to address these gaps by combining deep learning and edge computing to create a real-time, accurate, and scalable wildfire detection system.

---

## **Features**
1. **Model Benchmarking**:
   - Evaluate CNN models based on accuracy, F1-score, and inference time.
   - Fine-tune pre-trained models for wildfire detection tasks.

2. **Data Augmentation**:
   - Techniques include horizontal flips, rotations, Gaussian noise, and fog simulation.

3. **Deployment Feasibility**:
   - Optimize models for deployment on Raspberry Pi for real-world testing.

4. **Real-Time Inference**:
   - Achieves high-speed processing suitable for live data streams.

---

## **System Architecture**
![System Architecture](#)  
*Image Placeholder: Diagram of the wildfire detection pipeline, showing data input, preprocessing, training, and edge deployment.*

---

## **Setup Instructions**

### **1. Prerequisites**
- **Hardware Requirements for Training**:
  - Intel Core i7/AMD Ryzen 7 (or higher).
  - 16 GB RAM (32 GB recommended).
  - NVIDIA RTX 3060 (or higher).

- **Hardware Requirements for Edge Deployment**:
  - Raspberry Pi 5 (8 GB) or NVIDIA Jetson Nano.
  - Camera Module for real-time image capture.

- **Software Requirements**:
  - Python 3.8 or higher.
  - CUDA Toolkit & cuDNN for GPU acceleration.
  - PyTorch, NumPy, Albumentations, OpenCV, and Optuna.

### **2. Clone the Repository**
```bash
git clone https://github.com/your-username/wildfire-detection.git
cd wildfire-detection
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Prepare Dataset**
- Download the wildfire dataset from [Kaggle](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset).
- Unzip and organize into `data/` directory:
  ```
  data/
  ├── train/
  ├── val/
  └── test/
  ```

### **5. Model Training**
```bash
python train.py --model efficientnet --epochs 50 --batch_size 32
```

### **6. Model Evaluation**
```bash
python evaluate.py --model efficientnet --test_dir data/test
```

### **7. Deployment on Edge Devices**
- Convert the trained model to TensorFlow Lite or ONNX format:
  ```bash
  python convert_model.py --model_path saved_models/model.pth --format onnx
  ```
- Transfer the model to the Raspberry Pi and deploy using the `deploy.py` script.

---

## **Dataset**
The wildfire dataset used for this project is sourced from Kaggle. It consists of 2700 images split into:
- **Fire**: 1047 images
- **NoFire**: 1653 images  

Augmentation and preprocessing ensure robustness against real-world variations like lighting and smoke.

---

## **Model Training and Evaluation**
### **Training Configurations**
- **Baseline**: Pre-trained weights frozen, no data augmentation.
- **Fine-Tuned**: Selective layer unfreezing, hyperparameter optimization with Optuna.

### **Evaluation Metrics**
- Precision, Recall, F1-Score
- Inference time
- Edge device compatibility (memory and CPU usage)

---

## **Results**
| Model             | F1-Score (Baseline) | F1-Score (Fine-Tuned) | Inference Time (ms) |
|--------------------|---------------------|------------------------|---------------------|
| ConvNeXt-Tiny     | 91.22%             | 94.86%                | 37.1               |
| MobileNetV2       | 89.21%             | 93.54%                | 15.9               |
| EfficientNetB0    | 88.04%             | 93.65%                | 21.4               |
| ResNet18          | 87.61%             | 88.84%                | 16.9               |

---

## **Edge Deployment**
1. **Setup Raspberry Pi**:
   - Install Raspberry Pi OS.
   - Install dependencies:
     ```bash
     sudo apt-get install python3-opencv
     pip install tensorflow-lite
     ```

2. **Deploy the Model**:
   - Run the deployment script:
     ```bash
     python deploy.py --model_path model.tflite --input input.jpg
     ```

---

Here’s an expanded section for the **Future Scope** of your wildfire detection project, highlighting potential advancements and applications:



## **Future Scope**

### **1. Advanced Detection Techniques**
- **Multimodal Data Integration**: Combine multiple data sources, such as satellite imagery, drone feeds, and environmental sensors (e.g., temperature, humidity, wind speed), to enhance detection accuracy and provide richer context for fire prediction.
- **Hybrid Models**: Explore hybrid architectures that combine CNNs with other models, such as transformers or Support Vector Machines (SVMs), to improve the interpretability and robustness of the system.
- **Explainable AI (XAI)**: Incorporate techniques to make the AI model’s decisions interpretable for end-users, improving trust and enabling quicker responses.

---

### **2. Drone-Based Deployment**
- **Autonomous Drones**: Integrate wildfire detection models into drones equipped with thermal and optical cameras, enabling autonomous flight and monitoring of vast areas.
- **Swarm Technology**: Use multiple drones coordinated in real-time to survey larger regions, with data synchronized across edge devices for comprehensive coverage.
- **Dynamic Path Planning**: Equip drones with algorithms to adjust flight paths dynamically based on real-time fire detections or environmental conditions.

---

### **3. IoT and Edge Computing Enhancements**
- **IoT Integration**: Develop an IoT ecosystem where edge devices, drones, and centralized servers communicate seamlessly, enabling distributed detection and monitoring.
- **Energy-Efficient Models**: Optimize models further to reduce power consumption on edge devices, extending battery life and enabling longer operation periods for field-deployed systems.

---

### **4. Scalability and Deployment**
- **Global Scalability**: Design the system to adapt to different geographies, vegetation types, and climates by incorporating region-specific datasets and models.
- **Cloud-Native Infrastructure**: Utilize cloud platforms to enable scalable deployment and real-time synchronization across multiple regions for large-scale disaster management systems.
- **Urban Applications**: Adapt the system for urban fire detection, focusing on monitoring industrial areas, warehouses, and high-risk zones.

---

### **5. Enhanced Real-Time Monitoring**
- **Video Stream Analysis**: Optimize the models to process live video streams efficiently, increasing the system's ability to identify and localize fires in real time.
- **Event-Based Alerts**: Introduce intelligent alert systems that notify authorities based on detected fire severity and location, prioritizing high-risk situations.
- **Mobile App Integration**: Develop a mobile app for real-time alerts, allowing users to monitor fires, track drone footage, and receive safety recommendations.

---

### **6. Disaster Risk Management**
- **Predictive Analytics**: Incorporate predictive models to estimate the spread of wildfires based on current environmental conditions, enabling preemptive evacuation or firefighting strategies.
- **Historical Data Analysis**: Leverage past wildfire data to identify patterns, high-risk areas, and seasonality, supporting proactive disaster management policies.

---

### **7. Collaboration with Emergency Services**
- **Integrated Command Systems**: Collaborate with fire departments and disaster response teams to integrate the system into their workflows, providing actionable insights during emergencies.
- **Automated Resource Allocation**: Use AI to suggest optimal allocation of firefighting resources, such as water bombing, personnel deployment, and firebreak construction.

---

### **8. Advanced Analytics and Insights**
- **Behavioral Analysis**: Study the behavior of wildfires under different environmental conditions, improving the accuracy of predictions.
- **Heatmaps and Visualizations**: Develop intuitive dashboards that present real-time fire locations, spread predictions, and resource availability, aiding decision-makers.

---

### **9. Environmental and Climate Monitoring**
- **Carbon Emission Tracking**: Measure and report the impact of wildfires on carbon emissions, supporting global climate monitoring initiatives.
- **Biodiversity Monitoring**: Use drone footage and AI models to assess the impact of wildfires on local flora and fauna, aiding in ecological recovery efforts.

---

### **10. Community Engagement and Education**
- **Public Awareness Campaigns**: Provide communities with real-time fire risk levels and preventative measures through mobile and web apps.
- **Citizen Science**: Enable residents to upload fire-related images or videos, contributing to dataset growth and system accuracy.
- **Educational Tools**: Develop simulations and visualizations to educate the public and policymakers on wildfire behavior and the importance of early detection.
