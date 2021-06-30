# Mask Detection
A simple real-time mask-detection program capable of detecting masks on multiple faces through webcam input.


# Program Execution
To run the program, run mask_detection.py:
```sh
python3 mask_detection.py
```
# Custom Training (Optional)
To train the network yourself, create a directory named `data`. Inside this directory, create two directories `train` and `valid`. Within both of these directories, create two more directories named `mask` and `no-mask` respectively. Include the images to use for training within their respective directories.

You will also need to install the pandas library:
```sh
pip3 install pandas
```
You can train the model by running train.py:
```sh
python3 train.py
```

# Resources:
- Mask detection dataset: https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset 
