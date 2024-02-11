# Construction-of-LLMs
 Named Entity Recognition Master Thesis


# Named Entity Recognition on Business Transaction PDFs

## Overview

This thesis project focuses on addressing the challenges of Named Entity Recognition (NER) in business transaction descriptions, utilizing techniques like Fast Vocabulary Transfer (FVT) for model compression and adapting transformer-based encoder models like BERT and Roberta. The primary goal is to enhance NER accuracy in the specific context of business transactions, overcoming challenges related to diverse document layouts and structures.

## Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

Named entities play a crucial role in business transaction descriptions, yet existing NER models are often not tailored for this context. This thesis explores the application of NER techniques, with a specific focus on large language models (LLMs) and model compression using FVT. The project aims to provide valuable insights for NLP applications in the vertical domain.

## Methodology

The project utilizes transformer-based encoder models, specifically BERT and Roberta, obtained from Hugging Face. The adaptation of VIPI, termed FVT, is introduced for Fast Vocabulary Transfer. The methodology involves fine-tuning these models on a specialized business transaction dataset, exploring how FVT enhances model efficiency and performance.

## Dataset

The dataset comprises 10,000 business transaction PDFs, carefully preprocessed for NER tasks. The diversity of document layouts and structures in this dataset reflects real-world challenges. Pretraining on this domain-specific dataset is performed to create custom-trained models and tokenizers.

## Usage

To replicate the project, clone the GitHub repository, follow the provided instructions for setting up the environment, and run the code. Detailed guidelines for replicating the results, including dataset conversion, model training, and FVT implementation, can be found in the project's documentation.

## Results

The key findings highlight the effectiveness of FVT in reducing model size without significant performance compromise. The thesis provides comprehensive insights into the performance of adapted LLMs, emphasizing the practicality of FVT in domain-specific NLP applications.

## Contributing

Contributions to this project are welcome! Feel free to submit issues, feature requests, or pull requests. Please follow the guidelines outlined in the contribution documentation.

## License

This project is licensed under the [MIT License](LICENSE).
