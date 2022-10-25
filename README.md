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
## 2. Experiment Implementation
### 2.1 Software Implementation
Under python3.8 and PyTorch 11.1, 
the sparsity exploration experiment can be done by executing the following commands for Cora, PubMed, and Flickr datasets, respectively.
```
python python/Cora.py
python python/PubMed.py
python python/Flickr.py
```
The result of the experiment are put under ```python/log/```.

### 2.2 FPGA Implementation
To implement the FPGA design with HLS :   

1. download the dataset from  ```hls/data```
2. Create a new HLS project with test.cpp, test.h and weiht0.h as the source
3. Add the test_test.cpp as the test bench
4. Run C Simulation and Co-Simulation




