## model_3_cut.h5

Layers:  
``` python
model = keras.Sequential([
    keras.layers.Input((image_h, image_w, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64),
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
model.fit(ds_train, epochs=5, verbose=1)
```  
