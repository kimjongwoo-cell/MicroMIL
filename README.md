# MicroMIL: Advancing Cancer Diagnosis with Spatially Enhanced Multiple Instance Learning in Microscopy Imaging



### Overview

### Requirements
````
pip install -r requirements.txt 
````

### How to Run
````

conda create -n micromil python=3.9
source activate py39
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
````

# Execute MF2vec on Filmtrust dataset
````
python main.py
````

### Arguments

- `--lr`: Learning rate
  - **Description**: Specifies the learning rate used by the optimizer during training.

- `--weight_decay`: Weight decay
  - **Description**: Controls the amount of L2 regularization applied to the model's parameters to prevent overfitting.

- `--batch_size`: Batch size
  - **Description**: Determines the number of samples per batch used for training.

- `--cluster_number`: Cluster number
  - **Description**: Specifies the number of clusters or groups used in the model.

- `--epoch`: Number of epochs
  - **Description**: Sets the total number of complete passes through the dataset during training.

- `--patience`: Patience
  - **Description**: Number of epochs with no improvement after which training will be stopped if using early stopping.

- `--model_name`: Model name
  - **Description**: Name of the model architecture or type to be used (e.g., 'resnet18', 'regnet_y_400mf', etc.).

- `--seed`: Random seed
  - **Description**: Seed value used to initialize random number generators for reproducibility.

- `--layer`: Number of layers
  - **Description**: Number of layers in the neural network architecture.

- `--device`: Device
  - **Description**: Device (e.g., 'cuda:0', 'cpu') on which to run the model.

- `--shuffle`: Shuffle
  - **Description**: Flag indicating whether to shuffle the data during training.

- `--hidden_dim`: Hidden dimension
  - **Description**: Dimensionality of the hidden layers in the neural network.

- `--num_classes`: Number of classes
  - **Description**: Number of output classes or categories for the classification task.

- `--dropout_node`: Dropout rate for nodes
  - **Description**: Dropout rate applied to node features in the graph neural network.

- `--type`: Model type
  - **Description**: Type of the model architecture or framework used (e.g., 'graph', 'CNN', etc.).
