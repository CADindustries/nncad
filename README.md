<p align="center">
  <a href="https://robocadsim.readthedocs.io/en/latest/index.html">
    <img src="https://github.com/CADindustries/nncad/blob/main/nncad.png" alt="nncad logo" width="300" height="230">
  </a>
</p>
<h1 align="center">nncad</h1>

<h2>About:</h2>  
  
nncad is a project in which I create neural networks for WorldSkills competitions.  
  

## Keras, drawn traffic signs  
Train data: [Google drive](https://drive.google.com/drive/folders/1g7sy6le5bapZ45Z3536Fn7drX0kr9PiQ?usp=sharing)  
Validation data: [Google drive](https://drive.google.com/drive/folders/1RDAGq0F0SdDNE-b_s9e5fOSlf92nFLzU?usp=sharing)  
Models: [Google drive](https://drive.google.com/drive/folders/1RtRkPHqnLdS3mPfIjyr7MnOcwZGIaBIg?usp=sharing)

#### model_1  
Train data - ```loss: 0.0079 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```1.0```  
Weight - ```11 mb```  
Real time accuracy - ```None```  

#### model_2  
Train data - ```loss: 5.9177e-04 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```0.9900990099009901```  
Weight - ```225 mb```  
Real time accuracy - ```~30%```  

#### model_3  
Train data - ```loss: 0.0015 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```1.0```  
Weight - ```56 mb```  
Real time accuracy - ```None```  

#### model_4  
Train data - ```loss: 0.0022 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```1.0```  
Weight - ```28 mb```  
Real time accuracy - ```None```  

#### model_5  
Train data - ```loss: 0.0047 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```1.0```  
Weight - ```18 mb```  
Real time accuracy - ```~17%```  

#### model_6  
Train data - ```loss: 0.0499 - accuracy: 0.9812```  
Validation data using *sklearn.metrics.accuracy_score* - ```1.0```  
Weight - ```11 mb```  
Real time accuracy - ```None```  

#### model_7  
Train data - ```loss: 0.0166 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```1.0```  
Weight - ```18 mb```  
Real time accuracy - ```None```  

## Keras, drawn cut traffic signs  
Train data: [Google drive](https://drive.google.com/drive/folders/1Uj7YhqyDJiq5d2ufQbI_MXe7T2-fOlk9?usp=sharing)  
Validation data: [Google drive](https://drive.google.com/drive/folders/1ASyNqf_R8IEu-vuZCp2zQ8tsAzF4KrBj?usp=sharing)  
Models: [Google drive](https://drive.google.com/drive/folders/1RtRkPHqnLdS3mPfIjyr7MnOcwZGIaBIg?usp=sharing)

#### model_1_cut  
Train data - ```loss: 0.0000e+00 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```1.0```  
Weight - ```1 mb```  
Real time accuracy - ```~82%```  

#### model_2_cut  
Train data - ```loss: 0.0000e+00 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```1.0```  
Weight - ```4 mb```  
Real time accuracy - ```~85%```  

#### model_3_cut  
Train data - ```loss: 0.0000e+00 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```1.0```  
Weight - ```7 mb```  
Real time accuracy - ```~86%```  

#### model_4_cut  
Train data - ```loss: 0.0000e+00 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```1.0```  
Weight - ```14 mb```  
Real time accuracy - ```~89%```  

## Keras, real cut traffic signs  
Train data: [Google drive](https://drive.google.com/drive/folders/12xx6VoR_AA5yHHQANVNxE6SX6QAKS5zU?usp=sharing)  
Validation data: ```None```  
Models: [Google drive](https://drive.google.com/drive/folders/1RtRkPHqnLdS3mPfIjyr7MnOcwZGIaBIg?usp=sharing)

#### model_1_real   
Train data - ```loss: 0.0020 - accuracy: 0.9996```  
Validation data using *sklearn.metrics.accuracy_score* - ```None```  
Weight - ```30 mb```  
Real time accuracy - ```~87%```  

#### model_2_real   
Train data - ```loss: 0.0020 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```None```  
Weight - ```4 mb```  
Real time accuracy - ```~88%```  

#### model_2d1_real   
Train data - ```loss: 0.0012 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```None```  
Weight - ```950 kb```  
Real time accuracy - ```~87%```  
