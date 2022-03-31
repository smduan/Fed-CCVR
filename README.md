Implementation of the  paper [No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data](https://proceedings.neurips.cc/paper/2021/file/2f2b265625d76a6704b08093c652fd79-Paper.pdf)

Run this repo:

1. Download the cifar10 dataset and save as images in the dir "./data/"

    `python data_process.py`

2. Run the main procedure:

   `python main.py`

3. Run t-SNE visualization:

   `python visualize.py [--model_before_calibration MODEL_BEFORE_CALIBRATION] [--model_after_calibration MODEL_AFTER_CALIBRATION] [--random_state RANDOM_STATE] [--save_path SAVE_PATH]`

   Default arguments are:
   
   - `MODEL_BEFORE_CALIBRATION`: `./save_model/model-epoch9.pth`
   - `MODEL_AFTER_CALIBRATION`: `./save_model/model.pth`
   - `RANDOM_STATE`: `1`
   - `SAVE_PATH`: `./visualize/tsne.png`

