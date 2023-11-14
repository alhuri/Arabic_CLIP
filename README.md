# Arabic CLIP

## Model Details

Arabic CLIP is an adaptation of the Contrastive Language-Image Pre-training (CLIP) for the Arabic language. CLIP is an OpenAI-developed model that learns conceptual concepts from images and relates them with textual descriptions. This work attempts to improve the model's understanding and interpretation of visual information in the context of the Arabic language.


## Model Use


```python

from transformers import VisionTextDualEncoderModel, AutoTokenizer
model = VisionTextDualEncoderModel.from_pretrained("LinaAlhuri/Arabic-clip-bert-lit")
model.save_pretrained("arabic_clip") 

tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic", cache_dir=None, use_fast=True)

```



## Data

This was done through a combination of crawling Wikipedia and using commonly used pre-existing image datasets such as [CC](https://ai.google.com/research/ConceptualCaptions/). One of the most challenging obstacles for multimodal technologies is the fact that Arabic has few data resources, making huge dataset construction difficult. Another is the degradation of translated datasets adapted from well-known publicly available datasets. Whether the choice is to use translated data or genuine data, it is difficult to achieve the desired results depending on only one source, as each choice has its pros and cons. As a result, the goal of this work is to construct the largest Arabic image-text pair collection feasible by merging diverse data sources. This technique takes advantage of the rich information in genuine datasets to compensate for information loss in translated datasets. In contrast, translated datasets contribute to this work with enough pairs that cover a wide range of domains, scenarios, and objects.


| Dataset name | Images   |
| --- | --- |
|Arabic Conceptual Captions	|1,427,210|
|Arabic COCO 2014	|414,113|
|Arabic WIT	|109,366|
|Arabic Flicker8K	|24,272|
|Proposed (WAP) dataset	|151,252|
|Total	|2,126,213|



## Performance and Limitations

We have tested the efficacy of Arabic CLIP across different benchmarks tailored for tasks like zero-shot learning, image retrieval, localization, and image search.
- Conceptual Captions
- COCO
- ImageNet
- Unsplash

### Zero-shot Learning
###  Performance

| Multilingual CLIP            | Top 1   | Top 5   | Top 10  | Top 100 |
|------------------------------|---------|---------|---------|---------|
| **Short translation**        | 10.10   | 21.99   | 26.70   | 47.57   |
| **Long translation**         | 9.518   | 20.942  | 25.54   | 45.59   |

| LiT Arabic CLIP              | Top 1   | Top 5   | Top 10  | Top 100 |
|------------------------------|---------|---------|---------|---------|
| **Short translation**        | **18.66**   | **39.04**   | **47.38**   | **75.34**  |
| **Long translation**         | 16.43   | 36.36   | 45.09   | 73.96   |

### Image Retrieval
#### Conceptual Captions Evaluation

| Metric  | MCLIP | LiT Arabic CLIP |
|---------|-------|-----------------|
| **MRR@1** | 0.064  | 0.154           |
| **MRR@5** | 0.093 |  0.218           |
| **MRR@10** | 0.100 | 0.231           |

#### COCO Dataset Evaluation

| Metric  | MCLIP | LiT Arabic CLIP |
|---------|-------|-----------------|
| **MRR@1** | 0.043 | 0.063           |
| **MRR@5** | 0.068 | 0.099           |
| **MRR@10** | 0.074 | 0.108           |



## Limitations
To summarize the limitations into points
- Arabic CLIP struggles to count after 3.
- Limited genuine samples for the Arabic language.
- Various noises and biases might be introduced into Arabic CLIP because no studies have been conducted yet to address this issue in the published Arabic dataset or Arabic language models.

### Bias
For gender bias, it is important to note that Arabic uses a two-gender system in which all nouns are classified as masculine or feminine. 
However, this is not the case for English. Translating the text from English to Arabic may result in information loss or even make it prone to gender bias.



