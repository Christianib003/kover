# **Dataset Card for InsuranceQA V2**

This Dataset Card provides a structured summary of the `deccan-ai/insuranceQA-v2` dataset used in this project.


### **Dataset Details**

* **Dataset Name:** InsuranceQA Corpus (Version 2)
* **Hugging Face ID:** `deccan-ai/insuranceQA-v2`
* **Original Source:** [Insurance Library Website](https://www.insurancelibrary.com/)
* **Original Paper:** *Applying Deep Learning to Answer Selection: A Study and An Open Task* by Feng et al. (ASRU 2015)


### **Motivation**

* **Purpose:** The dataset was originally created to facilitate research in the domain-specific Question-Answering (QA) field, specifically for the task of "answer selection."
* **Project Goal:** For this project, we are adapting the dataset for a **generative QA task**. The goal is to train a model that can generate its own high-quality, free-text answers to insurance-related questions, rather than just selecting a pre-existing one.


### **Dataset Composition**

* **Data Source:** The questions are from real-world users, and the answers were written by professionals with domain expertise in the US insurance industry.
* **Content:** The dataset consists of pairs of `(input, output)`, where `input` is the user's question and `output` is the expert's answer.
* **Splits:** The data is officially pre-split into:
    * **Training Set:** 21,325 examples
    * **Validation Set:** 3,354 examples
    * **Test Set:** 3,308 examples


### **Preprocessing**

No preprocessing was required for this project. The data was loaded directly from the Hugging Face Hub in a clean, ready-to-use format. The original dataset curators handled the decoding from an `idx_...` format into plain text.


### **Limitations & Biases**

* **Geographical Bias:** The content is heavily focused on the **United States insurance market**. The information regarding policies, regulations, and terminology may not be applicable in other countries.
* **Temporal Relevance:** The data was collected around or before 2015. Insurance laws, products, and common practices may have changed since then, so some information could be outdated.
* **Domain Specificity:** The model trained on this data will be an expert in insurance but will have no knowledge of other domains.