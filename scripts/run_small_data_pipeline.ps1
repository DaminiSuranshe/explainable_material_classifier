Write-Host "Starting small-data pipeline"

Write-Host "Preparing dataset"
python -m scripts.prepare_dataset

Write-Host "Training model"
python -m src.training.train_small_data

Write-Host "Evaluating model"
python -m scripts.evaluate_small_data

Write-Host "Running Grad-CAM test"
python -m scripts.test_gradcam

Write-Host "Active learning suggestions"
python -m scripts.check_data_sufficiency

Write-Host "Pipeline complete"
