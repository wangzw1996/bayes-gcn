## 1. Environment Setup
The software experiment is implemented with Python 3.8 and pytorch11.1.   
The FPGA design is implemented Vitis HLS 2021.1
PyTorch and related packages can be installed using the following command.
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
pip install torch-geometric
```
