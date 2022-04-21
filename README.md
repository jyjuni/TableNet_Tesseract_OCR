# TableNet and OCR
Table Detection and Information Extraction based on TableNet and Tesseract. 

Implementation of Paper <i>TableNet: Deep Learning model for end-to-end Table detection and Tabular data extraction from Scanned Document Images</i>: https://arxiv.org/abs/2001.01469

## Data
### <b> Train data </b> 
We first train and test our model using the same Marmot dataset as the original TableNet Paper:
 
https://www.icst.pku.edu.cn/cpdp/sjzy/index.htm.

We also use the proposed column annotation in the paper. The annotated data with column labelling can be found in: 
https://drive.google.com/drive/folders/1QZiv5RKe3xlOBdTzuTVuYRxixemVIODp.


### <b> Transfer learning </b>

Following our previous research and experiment, we can then apply this pre-trained model on our FinTabNet dataset: 

https://developer.ibm.com/exchanges/data/all/fintabnet/. 

The annotation for FinTabNet dataset is originally in json. However, one may easily label and convert the labeled data using: 

https://github.com/tzutalin/labelImg 


## OCR
We use Tesseract for OCR, an open source text recognition (OCR) Engine, available under the Apache 2.0 license. It can be used directly, or (for programmers) using an API to extract printed text from images.:
https://github.com/tesseract-ocr/tesseract 

