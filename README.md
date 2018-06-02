# YOLOv3-caffe

usage: 
python detection_one.py --image=images/dog-cycle-car.png

### Acknowledgements

The major part is adapted from [Ayoosh Kathuria](https://github.com/ayooshkathuria)'s

amazing tutorial on YOLOv3 implementation in pytorch. 

The weight files can be downloaded from [csdn](https://download.csdn.net/download/jason_ranger/10452595).
(ignore the prototxt file. that's for caffe with interp layer, explanation is below)

### Notes

A simplest YOLOv3 model in caffe for python3.

This is merely a practice project. Note that I implemented an interp layer in python for compatibilty.

This is because interp layer is only viable in deeplab caffe, not in the official one. 

Moreover, the behavior of interp layer in deeplab is different from pytorch's UpsamplingBilinear2d,

in a sense that pytorch rescales (3,13,13) to (3,26,26) with a factor of 2, but deeplab caffe 

rescales it to (3,25,25) with the same factor. This causes weird performance degradation.

To fix this into a pure caffe model in C, you need to customize your interp layer.

In deeplab version, note in interp_layer.cpp, you need to fix the logic (take a example):

`height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1)`

into

`height_out_ = height_in_eff_ * zoom_factor`

