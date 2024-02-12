import tensorflow as tf
def SAC_UWNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 32 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = sa_conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, 1, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = sa_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, 2, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = sa_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, 3, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = sa_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate,4, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = sa_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate,5, batch_norm)

    # W-net layers
    upw_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    upw_16 = layers.concatenate([upw_16, conv_16], axis=3)
    up_convw_16 = sa_conv_block(upw_16, FILTER_SIZE, 8*FILTER_NUM, 0.2,6, batch_norm)

    poolw_8 = layers.MaxPooling2D(pool_size=(2,2))(up_convw_16)
    ct_16 = layers.concatenate([conv_8, poolw_8], axis=3)
    convw_16 = sa_conv_block(ct_16, FILTER_SIZE, 16*FILTER_NUM, dropout_rate,7, batch_norm)
    
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(convw_16)
    up_16 = layers.concatenate([up_16, up_convw_16], axis=3)
    up_conv_16 = sa_conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate,8, batch_norm)
    # UpRes 7
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = sa_conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate,9, batch_norm)
    # UpRes 8
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)
    up_conv_64 = sa_conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate,10, batch_norm)
    # UpRes 9
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = sa_conv_block(up_128, FILTER_SIZE, FILTER_NUM,dropout_rate,11, batch_norm)

    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, name='conv12', kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="SAC_UWNet")
    return model