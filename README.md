
# DSC180 - Models for Creditworthiness 

Machine Learning for credit score calculation. Python Version 3.9.5

## Description

In the complex landscape of financial services, creditworthiness is vital for managing financial risks, ensuring economic stability, and making reasonable lending decisions. The FICO score, the standard measure of creditworthiness, is updated depending on the creditor’s reporting schedule, typically ranging from one month to forty-five days. As a result, the FICO score fails to capture the rapid changes in an individual’s financial behavior or the emergence of new financial risks. Our technique aims to refine this process and update the score within two weeks, thereby significantly enhancing the efficiency and accuracy of creditworthiness assessment. 

## Installation

Before you can run the project, you need to install the necessary Python packages. This project depends on several third-party libraries, which are listed below:

- `numpy`: A fundamental package for scientific computing with Python.
- `pandas`: An open-source data analysis and manipulation tool.
- `torch`: An open-source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.
- `nltk`: A leading platform for building Python programs to work with human language data (Natural Language Toolkit).
- `transformers`: A library by Hugging Face offering state-of-the-art general-purpose architectures for Natural Language Processing (NLP).
- `scikit-learn`: A simple and efficient tools for predictive data analysis.

To install these packages, you can use `pip`, which is the package installer for Python. Simply run the following command in your terminal:

```bash
# Install required packages
pip install -r requirements.txt
```
Check if Python 3.9.5 is installed:

Open a terminal and run:

```bash
python3.9 --version
```

## Instruction

To begin working with this project, you'll need to set up your environment and obtain the necessary files and models. Follow these steps to get everything up and running:

1. **Clone the Repository and Download Pre-trained Model**
   - Use `git clone` to copy the project notebook and associated files to your local machine.
   - To access the dataset, unzip the `Transacation_outflows_3k.zip` file. You may request the password from wxia@ucsd.edu.
   - You can download the pre-trained model found [here](https://drive.google.com/file/d/1x_mEn9idwcN-vUSGziKpn4v9JexvTZJl/view?usp=sharing). Unzip the files into your cloned repository. Ensure that the binary files associated with the model are under the directory `/bank_transaction_model`.
2. Running the Notebook
   - If you wish to retrain the model with your data or tweak the training process, execute the entire notebook. This will take you through the entire pipeline from data preprocessing to model training.
3. Using the Fine-tuned Model for Inference
   - In case you want to skip the training phase and use the pre-trained model that you downloaded, first navigate to the **Fine Tuning** section and comment out all three cells. You may then execute the entire notebook.

### Begin execution from this point to use the model for making predictions or analyzing new data.

4. Important Note on Saving the Model
   - Be cautious not to run the command that saves the model more than once, as this will overwrite the previously saved model. If you've made significant changes and wish to save those, ensure you back up the original model or save the new one under a different name.




