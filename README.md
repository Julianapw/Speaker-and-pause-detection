# Speaker & Pause Detection in Phone Call Audio
 
## Description
This repository contains a pipeline developed in Python to detect pauses and identify different speakers in phone call audio recordings, using the Portuguese Speech Recognition Dataset. The project was developed in Google Colab and combines signal processing, speaker embeddings, unsupervised clustering, and automatic speech recognition (ASR) to answer three questions about a conversation: **when are the pauses**, **how many speakers are there**, and **who said what**.
 
**Goal:** Demonstrate an end-to-end speaker diarization pipeline for telephone-quality audio (8kHz), covering pause detection, speaker embedding extraction, clustering, label smoothing, and transcription — without relying on paid APIs.
 
## Features
- Audio loading and decoding from a Hugging Face dataset (`AudioDecoder` / `torchcodec`)
- Pause detection based on RMS energy in short time windows
- Classification of pauses into short pauses (within speech) vs. end-of-turn pauses
- Waveform visualization with highlighted pause regions
- Speaker voice embedding extraction using Resemblyzer
- Automatic detection of the number of speakers via clustering (KMeans + silhouette score)
- L2-normalized embeddings for cosine-like distance clustering
- Label smoothing to correct isolated misclassifications in short speech segments
- Speaker timeline visualization (waveform and bar-chart views, color-coded by speaker)
- Speech-to-text transcription per segment using OpenAI Whisper
- Final speaker-labeled dialogue output (text and tabular/DataFrame formats)
## Technologies & Resources Used
- Python
- Google Colab
- Hugging Face `datasets`
- NumPy
- Librosa
- Resemblyzer (speaker embeddings)
- scikit-learn (KMeans, AgglomerativeClustering, StandardScaler, PCA, silhouette score)
- OpenAI Whisper (speech-to-text)
- Pandas
- Matplotlib
- Portuguese Speech Recognition Dataset (`UniDataPro/portuguese-speech-recognition-dataset`)
## Pipeline Overview
1. **Load audio** from the dataset and extract raw samples and sample rate
2. **Detect pauses** using RMS energy thresholding and classify them as short pauses or end-of-turn pauses
3. **Extract speech segments** from the gaps between detected pauses
4. **Generate speaker embeddings** for each segment with Resemblyzer
5. **Cluster embeddings** to estimate the number of speakers and assign a speaker label to each segment
6. **Smooth speaker labels** to reduce noise from very short segments
7. **Transcribe** each segment with Whisper
8. **Combine** speaker labels and transcriptions into a chronological dialogue
