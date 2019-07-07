# Soccer Ball Detection using Deep CNN

# Usage
prepare datasets
```
python prepare_dataset.py
```
to reproduce best numbers
```
python test.py --reproduce=best
```
to reproduce all numbers
```
python test.py --reproduce=all
```
to evaluate on new data only to get output detection of the swetynet
```
python test.py --dataset=new_sweaty --data_root=/root_folder_of_dataset/
```
to evaluate on new sequence of data
```
python test.py --dataset=new_seq --data_root=/root_folder_of_dataset/
```
structure for the new data should be like in testDataset where each line of the ball.txt is relative path to the image, y center position, x center position, y resolution of image, x resolutionn of image

the result output of the network you can find in the folder 'seq_output'. The target heatmaps on the visualization consist of only zeros due to the implementation of the dataset. Make sure that the number of images in the new dataset is more than 20 if you use --dataset=new_seq.

## Implementation Details
```
py_models/joined_model.py
```


```
py_models/lstm.py
```

```
py_models/tcn_ed.py
```

```
py_train/evaluator.py
```

```
py_dataset/seq_dataset.py
```

```
arguments.py
```

### Reference

```
@inproceedings{Kukleva,
  title={Utilizing Temporal Information in Deep Convolutional Network for Efficient Soccer Ball Detection and Tracking },
  author={Kukleva, Anna and Asif Khan, Mohammad and Farazi, Hafez and Behnke, Sven},
  booktitle={Accepted for 23th RoboCup International Symposium, Sydney, Australia, to appear July 2019. },
  year={2019}
}
```
