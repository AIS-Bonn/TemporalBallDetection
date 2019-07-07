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
@inproceedings{schnekenburger2017detection,
  title={Detection and Localization of Features on a Soccer Field with Feedforward Fully Convolutional Neural Networks (FCNN) for the Adult-Size Humanoid Robot Sweaty},
  author={Schnekenburger, Fabian and Scharffenberg, Manuel and W{\"u}lker, Michael and Hochberg, Ulrich and Dorer, Klaus},
  booktitle={Proceedings of the 12th Workshop on Humanoid Soccer Robots, IEEE-RAS International Conference on Humanoid Robots, Birmingham},
  year={2017}
}
```
