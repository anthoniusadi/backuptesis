base_model_path = "/home/448690/ASL4/mobilenetv2.h5"
train_path='/home/m448690/ASL4/augmented_data/'
in_img=(224,224,3)
colname=["data","label"]
train_data = pd.read_csv('train_augmented.csv',dtype=str)
Y = train_data['label']
X = train_data['data']
kf = KFold(n_splits = 5,shuffle=True)
for train_index, val_index in kf.split(X):
    training_data = train_data.iloc[train_index]
    validation_data = train_data.iloc[val_index]
    base_model = tf.keras.applications.MobileNetV2(input_shape=in_img ,include_top=False, weights="mobilenetv2.h5")
    for layer in (base_model.layers):
        layer.trainable=True
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    x=Dense(1024,activation='relu')(x)
    x=Dense(812,activation='relu')(x)
    preds=Dense(10,activation='softmax')(x) 
    model=Model(inputs=base_model.input,outputs=preds)
    model.summary()
