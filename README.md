# Large-scale Visual Relationship Understanding

![alt text](https://github.com/facebookresearch/Large-Scale-VRD/blob/master/Examples.PNG)
<p align="center">Example results from the VG80K dataset.</p>

This is the Caffe2 implementation for [Large-scale Visual Relationship Understanding, AAAI2019](https://arxiv.org/abs/1804.10660).

Note: This code is for the VG80K dataset only. For results on VG200 and VRD please refer to the [PyTorch implementation](https://github.com/jz462/Large-Scale-VRD.pytorch).

## Caffe2

To install Caffe2 with CUDA support, follow the [installation instructions](https://caffe2.ai/docs/getting-started.html) from the [Caffe2 website](https://caffe2.ai/). **If you already have Caffe2 installed, make sure to update your Caffe2 to a version that includes the [Detectron module](https://github.com/pytorch/pytorch/tree/master/modules/detectron).**

Please ensure that your Caffe2 installation was successful before proceeding by running the following commands and checking their output as directed in the comments.

```
# To check if Caffe2 build was successful
python2 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# To check if Caffe2 GPU build was successful
# This must print a number > 0 in order to use Detectron
python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```

If the `caffe2` Python package is not found, you likely need to adjust your `PYTHONPATH` environment variable to include its location (`/path/to/caffe2/build`, where `build` is the Caffe2 CMake build directory).

## Other Dependencies

Install Python dependencies:

```
pip install numpy>=1.13 pyyaml>=3.12 matplotlib opencv-python>=3.2 setuptools Cython mock scipy
```

## Large-scale-VRD

Clone the Large-scale-VRD repository:

```
# Large-scale-VRD=/path/to/clone/Large-scale-VRD
git clone https://github.com/fairinternal/VRD $Large-scale-VRD
```

Set up Python modules:

```
cd $Large-scale-VRD/lib && make
```

## Annotations

Download VG annotation files from [here](https://www.dropbox.com/s/minpyv59crdifk9/datasets.zip). Put the zip file under `$Large-scale-VRD` and unzip it. You should see a `datasets` folder unzipped there.

## Datasets

Download VG80K images from [here](http://visualgenome.org/api/v0/api_home.html). Unzip all images into `$Large-scale-VRD/datasets/large_scale_VRD/Visual_Genome/images`.

## Pretrained Embedding Models

Download pretrained embeddings from [here](https://www.dropbox.com/s/r6uh5n9h76k41w7/Ji%20Zhang%20-%20embeddings.zip). Put the zip file under `$Large-scale-VRD/datasets/large_scale_VRD` and unzip it. You should see a "label_embeddings" folder and a "models" folders there.

## Our Trained Models

You can download our trained models from [here](https://www.dropbox.com/s/t5b1b2odn781035/checkpoints.zip). Put the zip file under `$Large-scale-VRD` and unzip it. You should see a `checkpoints` folder unzipped there.

## Training

To train VG80K with 8 GPU, run:

```
python tools/train_net_rel.py --cfg configs/vg/VG_wiki_and_relco_VGG16_softmaxed_triplet_2_lan_layers_8gpu.yaml
```

## Testing

To test VG80K with 8 GPU, run:

```
python tools/test_net_rel.py --cfg configs/vg/VG_wiki_and_relco_VGG16_softmaxed_triplet_2_lan_layers_8gpu.yaml 
```

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

Our revised annotations, linked above are based on Visual Genome which is licensed under:
[Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/). 
Our revised annotations are under Attribution-NonCommercial 4.0 International License which can be found under the LICENSE file in the root directory of this source tree.


## Citing Large-Scale-VRD
If you use this code in your research, please use the following BibTeX entry.
```
@conference{zhang2018large,
  title={Large-Scale Visual Relationship Understanding},
  author={Zhang, Ji and Kalantidis, Yannis and Rohrbach, Marcus and Paluri, Manohar and Elgammal, Ahmed and Elhoseiny, Mohamed},
  booktitle={AAAI},
  year={2019}
}
