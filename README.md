# Large-scale Visual Relationship Understanding

This is the Caffe2 implementation for Large-scale Visual Relationship Understanding, Ji Zhang, Yannis Kalantidis, Marcus Rohrbach, Manohar Paluri, Ahmed Elgammal, Mohamed Elhoseiny


## Caffe2

To install Caffe2 with CUDA support, follow the [installation instructions](https://caffe2.ai/docs/getting-started.html) from the [Caffe2 website](https://caffe2.ai/). **If you already have Caffe2 installed, make sure to update your Caffe2 to a version that includes the [Detectron module](https://github.com/caffe2/caffe2/tree/master/modules/detectron).**

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

## Datasets

1. Download datasets related files from [here](https://www.dropbox.com/s/minpyv59crdifk9/datasets.zip). Put the zip file under `$Large-scale-VRD` and unzip it. You should see a `datasets` folder unzipped there.

2. Download Visual Genome images from [here](http://visualgenome.org/api/v0/api_home.html). Unzip all images into `$Large-scale-VRD/datasets/large_scale_VRD/Visual_Genome/images`.

3. Download Visual Relationship Detection images from [here](https://cs.stanford.edu/people/ranjaykrishna/vrd/). Put training images into `$Large-scale-VRD/datasets/large_scale_VRD/Visual_Relation_Detection/train_images` and testing images into `$Large-scale-VRD/datasets/large_scale_VRD/Visual_Relation_Detection/test_images`.

## Pretrained Models

Download pretrained models from [here](https://www.dropbox.com/s/t5b1b2odn781035/checkpoints.zip). Put the zip file under `$Large-scale-VRD` and unzip it. You should see a `checkpoints` folder unzipped there.

## Training

To train Visual Genome with 1 GPU, run:

```
python tools/train_net_rel.py --cfg configs/vg/VG_wiki_and_relco_VGG16_softmaxed_triplet_2_lan_layers_1gpu.yaml
```

To train Visual Relationship Detection with 1 GPU, run:

```
python tools/train_net_rel.py --cfg configs/vrd/VRD_wiki_and_node2vec_VGG16_softmaxed_triplet_1gpu.yaml
```

## Testing

To test Visual Genome with 1 GPU, run:

```
python tools/test_net_rel.py --cfg configs/vg/VG_wiki_and_relco_VGG16_softmaxed_triplet_2_lan_layers_1gpu.yaml 
```

To test Visual Relationship Detection with 1 GPU, run:

```
python tools/test_net_rel.py --cfg configs/vrd/VRD_wiki_and_node2vec_VGG16_softmaxed_triplet_1gpu.yaml
```
