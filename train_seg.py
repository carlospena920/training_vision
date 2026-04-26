from ultralytics import settings, YOLO
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
# import os
# import mlflow

def main():
    # Enable MLflow
    settings.update({"mlflow": True})
    
    # Configure MLflow
    run_name = "Side_NOK_NylonInForm_Classified2"
    if not run_name:
      run_name = input("Enter run name (name of folder which includes data.yaml): ")

    # data = os.path.join("datasets", "rc1")
    data = f"datasets/{run_name}"

    # Train model
    model = YOLO("best.pt")  # Load a pretrained model (optional)
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
        patience=30,
 
        # Augmentations realistas
        scale=0.2,
 
        hsv_h=0.02,
        hsv_s=0.15,
        hsv_v=0.15,
    )

if __name__ == "__main__":
    main()