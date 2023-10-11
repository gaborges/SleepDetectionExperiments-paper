# ISABELA Sleep Analysis with federated learning
 
This file describes the how to reproduce the codes using data from ISABELA platform. 
It executes the codes using the tensorflow federated library, Python and Jupyter Notebook.

## Requirements

Windows or Linux with Conda, pip and Python 3.9+ support. 

GPU support is not required, but it greatly improves the performance. If you want to use it
consider the instructions in [link](https://www.tensorflow.org/install/pip?hl=pt-br).

Warning: The instructions will not work on macOS environment.

### Dataset

You must download the data from our [Kaggle Repository](https://www.kaggle.com/dsv/5804700) and add it to the folder Dataset_ECUADOR_2019.

### Miniconda

First, you will need Miniconda. If you did not have it, follow instructions according to your O.S.:
- Native Windows: https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html
- Linux or Windows with WSL 2.0: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

In Linux, you can download and install by the following commands:
- `curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh`
- `bash Miniconda3-latest-Linux-x86_64.sh`

Useful commands:
- Create a conda environment: `conda create --name tf python=3.9`
- If you need to remove the environment: `conda remove --name tf --all`
- Access the environment: `conda activate tf`
- Exit the environment: `conda deactivate`

### TensorFlow Federated libraries

The installations commands are:
- `pip install --upgrade tensorflow==2.10`
- `pip install --upgrade tensorflow-federated==0.40`
- `pip install flwr==1.4`
- `pip install flwr["simulation"]`

### Install support libraries to Jupyter Notebook

Execute the following command line: `pip install jupyter pandas scikit-learn matplotlib seaborn nltk beautifulsoup4 nest_asyncio imblearn`

### Execute jupyter notebook

`jupyter notebook`

## Execution instructions

The root of the project is composed of three folders:
- Dataset_ECUADOR_2019: Folder containing the raw ISABELA data you download from Kaggle.
- 00_scripts_raw_data: Folder data contains all the scripts to pre-process and run the federated learning codes.
- 00_scripts_user_filter_24hs_delete: Folder data contains all the scripts to pre-process and run the federated learning codes with the data filtering.

The both scripts folders are divided into four folders that must be executed based on the number order:
- [00_data_preprocessing](./scripts/00_data_preprocessing): it has the scripts to process the raw data. You must execute them all based on the order from the numbers there (from 00 to 06).
- [01_model_processing](./scripts/01_model_processing): it has the scripts to execute the machine learning process. After you execute the preprocessing scripts, execute all the files in the order of number (from 00 to 05).
- [02_data_analysis](./scripts/02_data_analysis): it has some data analysis and graphs scripts. You can copy the results from the model processing files and paste them there to summarize the data.
- [data_2019_processed](./scripts/data_2019_processed): it has the processed data generated from the preprocessing scripts.

Warning: the 01_model_processing script files **require very long times of processing**, so we strongly recommend exporting the notebook files to Python and execute them in a remote server with more processing resources. You can do it manually with the Jupyter Notebook web interface or by command lines.

An example of a command line to convert the Jupyter Notebook Files to Python is `jupyter nbconvert --to python notebook.ipynb` 

 

