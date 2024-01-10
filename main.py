from yolov3_model import make_yolov3_model
from Custom_Loss import yolo_loss
from WeightReader import WeightReader
from DataGenerator import DataGenerator

import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Conv2D
from keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
keras.backend.clear_session()


def main():
    #####################
    # Custom Parameters #
    #####################
    nr_layer="0"
    batch_size=2
    path="D:/Tobias/kvasir/"
    
    ids = os.listdir(path+"images/") 
    ######################
    # Load datagenerator #
    ######################
    generator_train = DataGenerator(path_main=path, list_id=ids[:800], batch_size=batch_size, data_augmentation=True)
    anchor = generator_train.get_anchor()
    generator_val = DataGenerator(path_main=path, list_id=ids[800:900], batch_size=batch_size, data_augmentation=False, anchor=anchor)
    
    ###################
    # Load base_model #
    ###################
    base_model = make_yolov3_model()
    weight_reader = WeightReader(path+'yolov3.weights')
    weight_reader.load_weights(base_model)
    
    #################################
    # Prepare Architecture of model #
    #################################
    pretrained_model = clone_model(base_model)
    nr_anchor = 1
    new_filter_size = 5 * nr_anchor

    for i in range(1,4):
        pretrained_model.layers[-i].filters = new_filter_size
        
    outp_82_old = pretrained_model.get_layer('leaky_80').output  
    outp_94_old = pretrained_model.get_layer('leaky_92').output 
    outp_106_old = pretrained_model.get_layer('leaky_104').output 
    
    outp_82_new = Conv2D(filters=new_filter_size, kernel_size=1, strides=1, padding='same', name='conv_81')(outp_82_old)
    outp_94_new = Conv2D(filters=new_filter_size, kernel_size=1, strides=1, padding='same', name='conv_93')(outp_94_old)
    outp_106_new = Conv2D(filters=new_filter_size, kernel_size=1, strides=1, padding='same', name='conv_105')(outp_106_old)
        
    main_model = Model(inputs=pretrained_model.input, outputs=[outp_82_new, outp_94_new, outp_106_new])
    
    #############################
    # Prepare Transfer Learning #
    #############################
    if nr_layer!="0":
        freeze = True
        for layer in main_model.layers:
            if layer.name=="conv_"+nr_layer:
                freeze=False
                
            if freeze==False:
                continue
            layer.trainable = False
        
    ##############################    
    # Prepare and start training #
    ##############################
    optimizer = optimizers.Adam(learning_rate=0.001)
    es = EarlyStopping(monitor="val_loss", patience=8)
    mcp = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_delta=10^(-7))
    
    main_model.compile(optimizer=optimizer, loss=yolo_loss)
    history = main_model.fit(generator_train,
                            validation_data=generator_val,
                            epochs=50,
                            batch_size=batch_size, 
                            callbacks=[mcp, reduce_lr, es])
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    main_model.save('models/last_model_test.h5')
    #keras.models.load_model("best_model.h5", custom_objects={'yolo_loss': yolo_loss})
    
if __name__=="__main__":
    main()
    
    
    
    
    
    
    