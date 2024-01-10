from keras import backend as K
## @brief custom loss function for yolov3 network. 
## @details This function can be passed to the fit() method as a loss function. 

def yolo_loss(y_true, y_pred):
    ##
    # @param y_true     Ground Truth of the training data.      
    # @param y_pred     Prediction of the training data.
    ##
    
    # Splitting the prediction/ground_truth tensor into different components   
    pred_confidence = K.sigmoid(y_pred[..., -1])
    true_confidence = y_true[..., -1]
    obj_mask = K.equal(true_confidence, 1)
    noobj_mask = K.equal(true_confidence, 0)
    
    obj_mask = K.cast(obj_mask, K.floatx())
    noobj_mask = K.cast(noobj_mask, K.floatx())

    true_xy = y_true[..., :2]
    pred_xy = y_pred[..., :2]
    true_wh = y_true[..., 2:4]
    pred_wh = y_pred[..., 2:4]
    
    # Binary crossentropy loss for objectness
    confidence_loss_obj = K.binary_crossentropy(true_confidence, pred_confidence) * obj_mask  
    confidence_loss_noobj = K.binary_crossentropy(true_confidence, pred_confidence) * noobj_mask
    
    # Mean squared error loss for bounding box coordinates
    xy_loss = K.mean(K.square(true_xy - pred_xy), axis=-1) * obj_mask
    wh_loss = K.mean(K.square(true_wh - pred_wh), axis=-1) * obj_mask
    
    # Combine the losses with appropriate weights
    lambda_coord = 6.0  # Weight for box coordinates
    lambda_noobj = 0.2  # Weight for confidence of non-objectness
    total_loss = (lambda_coord * K.sum(xy_loss) +
                  lambda_coord * K.sum(wh_loss) +
                  K.sum(confidence_loss_obj) +
                  lambda_noobj * K.sum(confidence_loss_noobj))

    return total_loss