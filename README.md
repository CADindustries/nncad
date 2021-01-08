# nncad
CAD's neural networks  

## Keras, drawn signs  
Train data: [Google drive](https://drive.google.com/drive/folders/1g7sy6le5bapZ45Z3536Fn7drX0kr9PiQ?usp=sharing)  
Validation data: [Google drive](https://drive.google.com/drive/folders/1RDAGq0F0SdDNE-b_s9e5fOSlf92nFLzU?usp=sharing)  
Models: [Google drive](https://drive.google.com/drive/folders/1RtRkPHqnLdS3mPfIjyr7MnOcwZGIaBIg?usp=sharing) or [GitHub](../main/models/keras_signs_drawn)

### model_1  
Train data - ```loss: 0.0079 - accuracy: 1.0000```  
Validation data using *sklearn.metrics.accuracy_score* - ```1.0```  
Layers:  
``` python
model = keras.Sequential([
    keras.layers.Input((image_h, image_w, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(3),
])
```  
Compiling:  
``` python
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[
          keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    ],
    metrics='accuracy'
)
```  
Fitting:  
``` python
model.fit(ds_train, epochs=10, verbose=1)
```  
