# fasterCV
Ensemble of data preprocessing, training tricks, logging, model testing, and model deployment as well as the directory structure to simple the  development of deep leaning with pytorch

# Data preprocessing
- lib/processing_utils.py:
  - face detection, landmarks detection，
  - video to frame, frame to face
  - data read and save, such as load_mat 
  - data analysis,

# Training,testing,and deployment
- lib/model_develop_utils.py 
  - example in src/xxx_train.py

- lib/convert_to_ddp.py convert conventional model to DistributedDataParallel model  
  - example in src/MnistPytorchDDP.py

- add model visualization with t-sen. 
  - see function sne_analysis in lib/model_develop_utils.py



# model of  state-of-the-art architectures
lib/model_arch_utils.py

mixup, warmup,cosine learning rate decay, logging, train, test and deploy model.


# Example

[https://github.com/shicaiwei123/MINIST_Test/tree/deeplearning_base_structure/pytorch](https://github.com/shicaiwei123/MINIST_Test/tree/deeplearning_base_structure/pytorch)


