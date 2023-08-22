# Analysis_Zebrafish
A framework for analyzing the calcium imaging of zebrafish

## Getting Started
```
git clone https://github.com/Knerlab/neural-activity-analysis.git
cd neural-activity-analysis
conda create --name zebrafish --file requirements.txt
conda activate zebrafish
```

**Dataset:** we release some of the datasets used in paper [here](). 

**Zebrafish Brain Atlas:** can be download [here](https://fishatlas.neuro.mpg.de/)

## Demo Example
```
python demo.py
```

## Data Preprocessing
```
python preprocess.py
```
Including the correction of motion artifacts, environmental lights artifacts, and signal normalizing

## ROI segmentation and signal extraction
```
python ROIExtraction.py
```
need to set the path of the atlas

## Dynamic Functional Connectivity Analysis
```
python meta_analysis_FC.py
```

## Spectra Analysis
```
python meta_analysis_fre_ROI.py
```
