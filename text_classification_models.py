import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, LSTM, Embedding,Dropout,SpatialDropout1D,Conv1D,MaxPooling1D,GRU,BatchNormalization
from tensorflow.keras.layers import Input,Bidirectional,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate,LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import numpy as np
# from tensorflow.nn import space_to_depth

class Models:

    embedding_dim = 500
    input_length = 100
    lstm_units = 75
    lstm_dropout = 0.4
    recurrent_dropout = 0.4
    spatial_dropout = 0.3
    filters = 32
    kernel_size = 3
    max_sequence_length = 75


    def build_Base_model(self,embedding_matrix):
        self.sequence_input = Input(shape=(self.max_sequence_length,))
        embedding = Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=self.max_sequence_length,
            trainable=False,
        )(self.sequence_input)
        base = SpatialDropout1D(self.spatial_dropout)(embedding)
        return base

    def build_GRU_model(self,base):
        base = GRU(self.lstm_units,dropout=self.lstm_dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=True)(base)
        base = GRU(self.lstm_units,return_sequences=True)(base)
        base = GRU(self.lstm_units,return_sequences=True)(base)
        base = GRU(self.lstm_units,return_sequences=True)(base)
        base = GRU(self.lstm_units,return_sequences=True)(base)
        base = GRU(self.lstm_units,dropout=self.lstm_dropout, recurrent_dropout=self.recurrent_dropout)(base)
        return base

    def build_LSTM_model(self,embedding_matrix):
        self.model = Sequential()
        self.model.add(
            Embedding(input_dim=embedding_matrix.shape[0],
                      output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix],
                      input_length=75,
                      trainable=False))
        self.model.add(SpatialDropout1D(0.5))
        self.model.add(Conv1D(self.filters, kernel_size=self.kernel_size,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Bidirectional(LSTM(self.lstm_units,dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
        self.model.add(SpatialDropout1D(0.5))
        self.model.add(Conv1D(self.filters, kernel_size=self.kernel_size,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Bidirectional(LSTM(self.lstm_units,dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
        self.model.add(SpatialDropout1D(0.5))
        self.model.add(Conv1D(self.filters, kernel_size=self.kernel_size,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Bidirectional(LSTM(self.lstm_units,dropout=0.5, recurrent_dropout=0.5)))
        self.model.add(Dense(4,activation='softmax'))
        self.model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy']) 

    def build_BLSTM_model(self,embedding_matrix):
        self.model = Sequential()
        self.model.add(
            Embedding(input_dim=embedding_matrix.shape[0],
                      output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix],
                      input_length=self.max_sequence_length,
                      trainable=False))
        self.model.add(SpatialDropout1D(0.5))
        self.model.add(Bidirectional(LSTM(self.lstm_units,return_sequences=True)))
        self.model.add(Bidirectional(LSTM(self.lstm_units, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(self.lstm_units,dropout=0.5, recurrent_dropout=0.5)))
        self.model.add(Dense(5,activation='softmax'))
        self.model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy']) 

    def build_BLTSM_model(self,base):
        base = Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                     dropout=self.lstm_dropout, recurrent_dropout=self.recurrent_dropout))(base)
        base = Bidirectional(LSTM(self.lstm_units, 
                 dropout=self.lstm_dropout, return_sequences=True, recurrent_dropout=self.recurrent_dropout))(base)
        base = Bidirectional(LSTM(self.lstm_units, recurrent_dropout=self.recurrent_dropout, return_sequences=True))(base)
        base = Bidirectional(LSTM(self.lstm_units, recurrent_dropout=self.recurrent_dropout,return_sequences=True))(base)
        base = Bidirectional(LSTM(self.lstm_units, recurrent_dropout=self.recurrent_dropout,return_sequences=True))(base)
        base = Bidirectional(LSTM(self.lstm_units, recurrent_dropout=self.recurrent_dropout,return_sequences=True))(base)
        base = Bidirectional(LSTM(self.lstm_units, 
                 dropout=self.lstm_dropout, recurrent_dropout=self.recurrent_dropout))(base)
        return base

    def build_BGRU_model(self,base):
        base = Bidirectional(GRU(self.lstm_units, return_sequences=True,
                     dropout=self.lstm_dropout, recurrent_dropout=self.recurrent_dropout))(base)
        base = Bidirectional(GRU(self.lstm_units, 
                 dropout=self.lstm_dropout, return_sequences=True, recurrent_dropout=self.recurrent_dropout))(base)
        base = Bidirectional(GRU(self.lstm_units, recurrent_dropout=self.recurrent_dropout, return_sequences=True))(base)
        base = Bidirectional(GRU(self.lstm_units, recurrent_dropout=self.recurrent_dropout,return_sequences=True))(base)
        base = Bidirectional(GRU(self.lstm_units, recurrent_dropout=self.recurrent_dropout,return_sequences=True))(base)
        base = Bidirectional(GRU(self.lstm_units, recurrent_dropout=self.recurrent_dropout,return_sequences=True))(base)
        base = Bidirectional(GRU(self.lstm_units, 
                 dropout=self.lstm_dropout, recurrent_dropout=self.recurrent_dropout))(base)
        return base
        
    def build_LTSM_model(self,base):
        base = LSTM(self.lstm_units, return_sequences=True,
                     dropout=self.lstm_dropout, recurrent_dropout=self.recurrent_dropout)(base)
        base = LSTM(self.lstm_units, 
                 dropout=self.lstm_dropout, return_sequences=True, recurrent_dropout=self.recurrent_dropout)(base)
        base = LSTM(self.lstm_units, recurrent_dropout=self.recurrent_dropout,return_sequences=True)(base)
        base = LSTM(self.lstm_units, recurrent_dropout=self.recurrent_dropout,return_sequences=True)(base)
        base = LSTM(self.lstm_units, recurrent_dropout=self.recurrent_dropout,return_sequences=True)(base)
        base = LSTM(self.lstm_units, recurrent_dropout=self.recurrent_dropout,return_sequences=True)(base)
        base = LSTM(self.lstm_units, 
                 dropout=self.lstm_dropout, recurrent_dropout=self.recurrent_dropout)(base)
        return base

    def build_CNN_model(self,base):
        base = Conv1D(self.filters, kernel_size=self.kernel_size,kernel_regularizer=regularizers.l2(0.0005), padding='same')(base)
        base = LeakyReLU(alpha=0.2)(base)
        base = Conv1D(self.filters, kernel_size=self.kernel_size,kernel_regularizer=regularizers.l2(0.0005), padding='same')(base)
        base = LeakyReLU(alpha=0.2)(base)
        base = Conv1D(self.filters, kernel_size=self.kernel_size,kernel_regularizer=regularizers.l2(0.0005), padding='same')(base)
        base = LeakyReLU(alpha=0.2)(base)
        base = Conv1D(self.filters, kernel_size=self.kernel_size,kernel_regularizer=regularizers.l2(0.0005), padding='same')(base)
        base = LeakyReLU(alpha=0.2)(base)
        base = Conv1D(self.filters, kernel_size=self.kernel_size,kernel_regularizer=regularizers.l2(0.0005), padding='same')(base)
        base = LeakyReLU(alpha=0.2)(base)
        avg = GlobalAveragePooling1D()(base)
        max = GlobalMaxPooling1D()(base)
        concat = concatenate([avg,max])
        base = BatchNormalization()(concat)
        return base

    def f1(self,y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.
            Only computes a batch-wise average of recall.
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.
            Only computes a batch-wise average of precision.
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    def weighted_categorical_crossentropy(self,weights):
    
        weights = K.variable(weights)
            
        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss
        
        return loss