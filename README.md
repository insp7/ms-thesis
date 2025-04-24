This repository references code from [EEGViT](https://github.com/ruiqiRichard/EEGViT). References Cited in code comments.

File Descriptions:

check_task1_connection_with_task2.ipynb - explored if learning representations from task 1 help on task 2.

downsample_data.ipynb - just a helper file to take the source .npz data file and downscale it to your desired percentage.

EEGViT_TCN_Modified - experimented by modifying with kernal and dropout sizes.

SwinTransformerBaseline.ipynb - experimented with basic SwinTrasnformer (non-pretrained version). Potential performance improvements that may come with a pretrained version can be tested in the future.

Task2_direction_task_pre_training.ipynb - Pretraining on Task 2 as a task 3 problem.

Task3_absolute_position_fine_tuning.ipynb - Fine-tuning task 3 on the pretrained ViT-encoder weights from task 2.

TrainAndError.py - Trying out new ideas/models.
