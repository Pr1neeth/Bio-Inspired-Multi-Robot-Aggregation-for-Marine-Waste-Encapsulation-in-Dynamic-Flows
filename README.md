# Bio-Inspired-Multi-Robot-Aggregation-for-Marine-Waste-Encapsulation-in-Dynamic-Flows
This project explores Dynamic Encapsulation, a collaborative multi-robot behavior in which a team of robots learns to locate, surround, and track drifting targets. We apply this behavior to a real-world environmental challenge: protecting marine ecosystems from ocean plastic pollution
While single-robot systems struggle with limited coverage and poor fault tolerance, a multi-robot system (MRS) provides the scalability and robustness needed to monitor large ocean regions. In this work, a learning-based MRS is trained to:

#### 1. Detect and navigate toward floating debris patches
#### 2. Form a stable circular “encapsulation” around the debris
#### 3. Maintain the formation as the patch drifts over time

The goal is to create a robotic “shield” that protects marine life from hazardous debris until cleanup efforts can safely remove it.

## Installation

### 1. Clone the repo

```
git clone https://github.com/Pr1neeth/Bio-Inspired-Multi-Robot-Aggregation-for-Marine-Waste-Encapsulation-in-Dynamic-Flows
```
### 2. Create a conda env (optional)
```
conda create -n ocean python=3.10
conda activate ocean
```
### 3. Install Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
### 3. Run Static Trained Model
```
cd Static
python3 run_trained_ppo_static.py
```

### 4. Train a Static Model
```
cd Static
python3 train_ppo_static.py --timestep 1000000 --vec 8 --seed 0
```

### 5. Run Trained Dynamic Model
```
cd Dynamic
python3 run_trained_ppo_dynamic.py
```
### 6. Train a Dynamic Model
```
cd Dynamic
python3 train_ppo_dynamic.py --timestep 1000000 --vec 8 --seed 0

