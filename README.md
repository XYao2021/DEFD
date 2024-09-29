# DEFD Codes
The official Pytorch code for DEFD-SGD algorithm. These codes only use for experiments in paper "Compressed Decentralized Learning With Error Feedback Under Data Heterogeneity" for ICLR 2025.

# Python Libraries
The required libraries can be found in "requirements.txt"

# Configures
The configuration parameters can be found in "config.py".

# Datasets
Datasets include FashionMNIST and CIFAR10. The other datasets can be applied if add the python file for that dataset under "dataset" folder.

# Models
The models are custimized model for FashionMNIST and CIFAR10. The models can be changed using files under "model" folder.

# Running
To run the experiments, using "python -main.py" command to run the file called "main.py" in the command line. The configurations can be changed by specifying the configuration parameters in the command line after "main.py", the details can be find in file "config.py". The other useful functions can be find in "util.py" file under "util" folder.
