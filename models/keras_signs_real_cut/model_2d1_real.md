## model_2d1_real.h5

Image shape: ```1 x 30 x 40 x 1```

Layers:  
``` python
model = keras.Sequential([
    keras.layers.Input((image_h, image_w, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='sigmoid'),
    keras.layers.Dense(32, activation='sigmoid'),
    keras.layers.Dense(3, activation='sigmoid'),
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
model.fit(ds_train, epochs=15, verbose=1)
```  
