# LLM-GPT-2-chatbot
This project is a large language model (LLM) with the same architecture of the original GPT-2 model developed by openAI, coded from scratch. The model was further fine-tuned with an instruction dataset of the alpaca input-instruction-response format to further train it into being an interactive chatbot.

## Project Goal:

The project aims to fine-tune a pre-trained GPT-2 language model on a dataset of instructions and responses, enabling it to generate relevant and coherent responses to new instructions. This involves data preparation, model loading, fine-tuning, evaluation, and response extraction.

## Project Steps:

### Data Preparation:

- Dataset Loading: The project begins by loading a dataset of instructions and responses in JSON format. The dataset contains entries of the alpaca format (a standard format used to train LLM models) with the fields "instruction", "input" and "response" 

- Data Formatting: convert the instructions and inputs into the alpaca format (a Dataset format that was used to train Llama 3.1 models). This format includes delimiters/special tokens to separate instructions and responses.

- Data Splitting: The dataset is split into training, validation, and test sets to facilitate model training and evaluation.

- Batching and Tokenization: The data is organized into batches using functions. These functions handle padding, truncation, and creation of input-target pairs for training the language model. The tiktoken library (Byte pair encoding) is used for tokenization, converting text into numerical representations.

### Model Loading and Initialization:

- Pre-trained Model weights Selection: A pre-trained GPT-2 model is selected as the base for fine-tuning. The project utilizes requests to download publicaly available weights of the GPT-2 model to fit them into the model.

- Model Architecture: A custom GPT model architecture (GPTModel) is defined, inheriting from the nn.Module class in PyTorch. This architecture includes embedding layers, transformer blocks, and a final output layer.

- Weight Loading: The pre-trained weights are loaded into the custom GPT model. This initializes the model with knowledge learned from a large text corpus.

### Model Fine-tuning:

- Loss Calculation: Functions are defined to calculate the loss during training. Cross-entropy loss is used to measure the difference between the model's predictions and the true targets.

- Training Loop: The training function orchestrates the training process. It iterates over epochs and batches, calculates the loss, performs backpropagation to compute gradients, and updates the model's parameters using the AdamW optimizer.

- Evaluation: During training, the model's performance is periodically assessed on the training and validation datasets. This helps monitor progress and detect potential overfitting.

### Response Extraction and Saving:

- Text Generation: training data is fed to the fine tuned model to generate text from the fine-tuned model given an input instruction.

- Response Extraction: The generated text is processed to extract the relevant response portion, removing any formatting or extraneous information.

- Saving Responses: The extracted responses are stored back in a list and saved to a JSON file for later analysis or use.

### Model Testing and Evaluation:

- Sample Generation: custom inputs are fed into the model to generate text samples from the fine-tuned model, demonstrating its capabilities.

- Loss Visualization: The training and validation losses are plotted to visualize the learning progress.

- Qualitative Evaluation: The generated responses are examined qualitatively to assess their coherence, relevance, and overall quality.

Overall:

This project follows a standard workflow for fine-tuning language models. It involves careful data preparation, leveraging a pre-trained model, fine-tuning on a specific task, evaluating performance, and extracting the desired outputs. The project demonstrates how to adapt a powerful language model like GPT-2 to generate responses to instructions, showcasing its potential for various natural language processing applications.

