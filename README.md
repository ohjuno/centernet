# Objects as Points

Implementation of "Objects as Points", in PyTorch.

The official repository is here, thanks fot the great work!
> [**Objects as Points**](https://github.com/xingyizhou/CenterNet)  
> Xingyi Zhou, Dequan Wang, Philipp Kr&auml;henb&uuml;hl  
> *arXiv technical report ([arXiv 1904.07850](http://arxiv.org/abs/1904.07850))*  


## What's Different?

The models in the original repository are also small, fast, and accurate, but **not enough** for CPU and mobile devices.  

Therefore, I want to replace the backbone of the model with a variant of MobileNet and tune the operations to make them suitable for CPU inference.
I also change the heatmap loss function to compensate for the accuracy lost in the lightweighting process.

## ONNX Export

All models can be converted to the ONNX format. For more information, see [export.py](export.py)

## License

Like the original, this repository is released under the MIT License (see [LICENSE](LICENSE) for details).
However, some code is borrowed from [torchvision](https://github.com/pytorch/vision) (mobilenet v3 implementation).
Please refer to the original license of there projects.
