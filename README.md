# InsuranceQA Generative Chatbot

This project is a domain-specific chatbot designed to answer questions about insurance policies. It is built by fine-tuning a pre-trained Transformer model on the InsuranceQA V2 dataset.

## Technical Choices

### **1. Approach: Generative vs. Extractive QA**

The assignment instructions mention two primary approaches: extractive and generative question-answering. While extractive QA (selecting an answer from a given context) is a valid method, this project intentionally follows the assignment's recommendation to explore **generative QA**.

The generative approach allows the chatbot to formulate its own answers in a flexible, free-text format. This results in a more natural and dynamic user experience, as the model is not limited to a fixed set of pre-written responses. This path better demonstrates the advanced capabilities of modern Transformer models.

### **2. Model Selection: T5 (Text-To-Text Transfer Transformer)**

For the generative model, we have selected **T5**, specifically the `t5-small` variant, for several key reasons:

* **Natural Fit for QA:** T5 is pre-trained on a "text-to-text" framework. It treats every NLP problem as a task of converting an input text into a new output text. This is a perfect match for our question-answering goal, where we give the model a question (input text) and expect an answer (output text).
* **Proven Performance:** T5 has demonstrated strong performance across a wide range of generative tasks, including summarization, translation, and question-answering.
* **Efficiency:** The `t5-small` variant provides a great balance between performance and computational efficiency, making it feasible to fine-tune on standard hardware.