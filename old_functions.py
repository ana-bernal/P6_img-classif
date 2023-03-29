def augmentation_layers(input_shape):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
         
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.RandomCrop(image_size[0], image_size[1]),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
        ]
    ) 

    # model.compile()

    return model


# my_model = augmentation_layers(image_size + (3,))
# keras.utils.plot_model(my_model, show_shapes=True)

def make_cnn_model(input_shape,filters, ker_sizes): # input_shape = image_size + (3,)

    # Model
    model = keras.Sequential(
        [   
            # keras.Input(shape=input_shape),     
            augmentation_layers(input_shape),    

            # Rescaling
            layers.Rescaling(1./255),

            # Block 1
            layers.Conv2D(filters=filters[1],
                          kernel_size=ker_sizes[1],
                          strides=(1,1),
                          padding="same"),  # for having outputsize=inputsize
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            # Block 2
            layers.Conv2D(filters=filters[2],
                          kernel_size=ker_sizes[2],
                          strides=(1,1),
                          padding="same"),  # for having outputsize=inputsize
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            # # Block 3
            # layers.Conv2D(filters=filters[3],
            #               kernel_size=ker_sizes[3],
            #               strides=(1,1),
            #               padding="same"),  # for having outputsize=inputsize
            # layers.BatchNormalization(),
            # layers.Activation("relu"),
            # layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            # Block 4
            # layers.Conv2D(filters=filters[4],
            #               kernel_size=ker_sizes[4],
            #               strides=(1,1),
            #               padding="same"),  # for having outputsize=inputsize
            # layers.BatchNormalization(),
            # layers.Activation("relu"),
            # layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            # # Block 5
            # layers.Conv2D(filters=filters[5],
            #               kernel_size=ker_sizes[5],
            #               strides=(1,1),
            #               padding="same"),  # for having outputsize=inputsize
            # layers.BatchNormalization(),
            # layers.Activation("relu"),
            # layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            # Block 6
            layers.Flatten(),
            # layers.Dense(units[1], activation="relu"),
            layers.Dropout(0.3),
            # layers.Dense(units[2], activation="relu"),
            # layers.Dropout(0.5),

            layers.Dense(15, activation="softmax")
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model