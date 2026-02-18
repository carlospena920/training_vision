python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt


ls datasets/Best_Seg_DefSide021326/images/train | head -39 | while read img; do
    mv datasets/Best_Seg_DefSide021326/images/train/$img \
       datasets/Best_Seg_DefSide021326/images/val/

    label="${img%.png}.txt"

    if [ -f datasets/Best_Seg_DefSide021326/labels/train/$label ]; then
        mv datasets/Best_Seg_DefSide021326/labels/train/$label \
           datasets/Best_Seg_DefSide021326/labels/val/
    fi
done