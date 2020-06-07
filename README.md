## Steps to use

Clone this repository

```bash
git clone https://github.com/gnoya/face_recognition.git
```

Move to its root folder:

```bash
cd face_recognition
```

Install the required packages (may take a while):

```bash
pip3 install -r requirements.txt
```

Go to weights directory:

```bash
cd ./yolov3/weights
```

Download the weights:

```bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xYasjU52whXMLT5MtF7RCPQkV66993oR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xYasjU52whXMLT5MtF7RCPQkV66993oR" -O yolov3_face.weights && rm -rf /tmp/cookies.txt
```

Go back to the root of this repository:

```bash
cd ../../
```

Run the face recognition

```bash
python3 main.py
```

## Credit to:

YOLOv3 PyTorch: https://github.com/eriklindernoren/PyTorch-YOLOv3

YOLOv3 weights and cfg: https://github.com/sthanhng/yoloface
