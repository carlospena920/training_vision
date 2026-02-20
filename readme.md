python -m venv venv

.\venv\Scripts\activate

python -m pip install --upgrade pip

pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121



pip install -r requirements.txt


ls datasets/Best_Seg_DefSide021326/images/train | head -110 | while read img; do
    mv datasets/Best_Seg_DefSide021326/images/train/"$img" \
       datasets/Best_Seg_DefSide021326/images/val/
done

python .\train_seg.py