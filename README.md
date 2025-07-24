cd pro_hand-detection
python -m venv .
source Scripts/activate
pip install -r requirements.txt

python utils/preprocess.py
python train_model.py
python detect_realtime.py
