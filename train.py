from imports import *
from attentionLayer import AttentionLayer
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json

from configparser import ConfigParser
config = ConfigParser()
config.read('config_var.ini')
max_len_text = config.getint('text_len', 'max_len_text')
max_len_summary = config.getint('text_len', 'max_len_summary')
EMBEDDING_DIM = config.getint('dimensions', 'EMBEDDING_DIM')
latent_dim = config.getint('dimensions', 'latent_dim')

K.clear_session() 
# latent_dim = 300
latent_dim = 100 

def build_model(embedding_matrix, x_voc_size, y_voc_size):
    # Encoder 
    encoder_inputs = Input(shape=(max_len_text,)) 
    enc_emb = Embedding(x_voc_size, 100, weights=[embedding_matrix],input_length=max_len_text, trainable=False)(encoder_inputs)
    # enc_emb = Embedding(len(x_tokenizer.word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
    #       input_length=max_len_text, trainable=False)(encoder_inputs) 

    #LSTM 1 
    encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True) 
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb) 

    #LSTM 2 
    encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True) 
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1) 

    #LSTM 3 
    encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True) 
    encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2) 

    # Set up the decoder. 
    decoder_inputs = Input(shape=(None,)) 
    dec_emb_layer = Embedding(x_voc_size, 100, weights=[embedding_matrix],input_length=max_len_text, trainable=False) 

    dec_emb = dec_emb_layer(decoder_inputs) 

    #LSTM using encoder_states as initial state
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) 
    decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c]) 

    #Attention Layer
    attn_layer = AttentionLayer(name='attention_layer') 
    # attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs]) 
    # attn_out = tf.keras.layers.Attention()([encoder_outputs, decoder_outputs])
    attn_out = tf.keras.layers.Attention()([decoder_outputs, encoder_outputs]) # Working but not in inference

    # Concat attention output and decoder LSTM output 
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    #Dense layer
    decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax')) 
    decoder_outputs = decoder_dense(decoder_concat_input) 

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
    model.summary()

    plot_model(model, to_file='./output/plot_model.png', show_shapes=True,show_layer_names=True)

    return model, encoder_inputs, encoder_outputs, state_h, state_c, dec_emb_layer, decoder_inputs, decoder_lstm, attn_layer, decoder_dense


def train_model(model, X_train, y_train, X_test, y_test):
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
    history=model.fit([X_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:],
                  epochs=4,callbacks=[es],batch_size=128,
                  validation_data=([X_test,y_test[:,:-1]],
                  y_test.reshape(y_test.shape[0],y_test.shape[1], 1)[:,1:]))
        
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plt.show()
    plt.savefig('./output/Visualizing_accuracy .png')
    
def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("./model/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./model/model.h5")
    print("Saved model to disk")
 
# later...

def load_model():
    # load json and create model
    json_file = open('./model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./model/model.h5")
    print("Loaded model from disk")
    
    return loaded_model
    
    
    