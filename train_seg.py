from ultralytics import settings, YOLO
# import os
# import mlflow

def main():
    # Enable MLflow
    settings.update({"mlflow": True})
    
    # Configure MLflow
    run_name = "Best_Seg_DefSide021326"
    if not run_name:
      run_name = input("Enter run name (name of folder which includes data.yaml): ")

    # data = os.path.join("datasets", "rc1")
    data = f"datasets/{run_name}/data.yaml"

    # Train model
    model = YOLO("yolo26s --seg.pt")
    results = model.train(
        device=0,
        data=data,
        epochs=300,
        batch=8,
        imgsz=640,
        name=run_name,
        deterministic=True,
 
        # Optimización
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        freeze=0,
        patience=50,
 
        # Augmentations realistas
        degrees=10,
        translate=0.03,
        scale=0.2,
        shear=0.0,
 
        hsv_h=0.02,
        hsv_s=0.15,
        hsv_v=0.15,
    )

if __name__ == "__main__":
    main()