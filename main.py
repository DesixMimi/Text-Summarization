import sys
from imports import *
from functions import *
from train import *
from inference import *

def main(param=0):
    
    ''' Loading data (param=0 will load the data from local file,
    param=1 will download the data from tf)         -------------------------------------------------
    '''
    
    # This will load the data directly from tf and create df
    if (param == 1):
        df = create_df_from_tf(200)
        df_to_csv(df)
    
    else:
        df = csv_to_df()
    
    df.head()
    
    i = 1
    print("--- Full text ---")
    print(df["text"][i])
    print("--- Summary ---")
    print(df["y"][i])
    
    ''' Data cleaning, preprocessing and analysis   -----------------------------------------------
    '''
    ## create stopwords
    lst_stopwords = nltk.corpus.stopwords.words("english")
    ## add words that are too frequent
    lst_stopwords = lst_stopwords + ["cnn","say","said","new"]
    
    # apply function to both text and summaries
    df["text_clean"] = df["text"].apply(lambda x: utils_preprocess_text(x, punkt=True, lower=True, slang=False,
                                                                        lst_stopwords=lst_stopwords, stemm=False, lemm=False))
    df["y_clean"] = df["y"].apply(lambda x: utils_preprocess_text(x, punkt=True, lower=True, slang=False,
                                                                  lst_stopwords=lst_stopwords, stemm=False, lemm=False))
    
    word_count_plot(df)
    # print_clean_text(df)
    
    # Selecting the news and summaries whose length falls below or equal to max_len_text and max_len_summary
    # Removing outliers
    df_short = truncate_text_length(df)
    
    # Applying Start and End tokens
    df_short["y_clean"] = df_short["y_clean"].apply(lambda x : 'sostok '+ x + ' eostok')
    print("split")
    X_train, X_test, y_train, y_test = train_test_split(np.array(df_short['text_clean']),np.array(df_short['y_clean']),
                                                        test_size= 0.1 ,random_state=42,shuffle=True)
    
    ''' Tokenization & GloVe Embedding   -----------------------------------------------
    '''

    X_train, X_test, x_voc_size, x_tokenizer = X_y_tokenization(X_train, X_test)
    y_train, y_test, y_voc_size, y_tokenizer = X_y_tokenization(y_train, y_test)
    
    print("Text vocab size:",x_voc_size)
    print("Summarization vocab size:",y_voc_size)
    print(X_train.shape) 
    print(y_train.shape)
    
    # checking whether word count of start token is equal to length of the training data
    y_tokenizer.word_counts['sostok'],len(y_train)

    embeddings_index = glove_100d_dictionary()
    embedding_matrix, word_index = map_word_to_glove(x_tokenizer, embeddings_index)
    
    print_embd_explain(embedding_matrix,embeddings_index, word_index)
    
    print(f"We have {X_train.shape[0]} examples of training text, each example has {X_train.shape[1]} features/dimension")
    print(f"We have {y_train.shape[0]} examples of training summaries, each example has {y_train.shape[1]} features/dimension")
    print("\n")
    print(f"We have {X_test.shape[0]} examples of testing text, each example has {X_test.shape[1]} features/dimension")
    print(f"We have {y_test.shape[0]} examples of testing summaries, each example has {y_test.shape[1]} features/dimension")
    
    embedding_layer = Embedding(len(x_tokenizer.word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                input_length=max_len_text, trainable=False,  name='embedding_layer')
    
    print("building the model")
    model, encoder_inputs, encoder_outputs, state_h, state_c, dec_emb_layer, decoder_inputs, decoder_lstm, attn_layer,decoder_dense = build_model(embedding_matrix, x_voc_size, y_voc_size)
    
    train_model(model, X_train, y_train, X_test, y_test)
    save_model(model)
    
    # model = load_model()
    
    # build the dictionary to convert the index to word for target and source vocabulary
    reverse_target_word_index=y_tokenizer.index_word 
    reverse_source_word_index=x_tokenizer.index_word 
    target_word_index=y_tokenizer.word_index
    
    
    encoder_model, decoder_model = build_inference_model(model, encoder_inputs, encoder_outputs, state_h, state_c, dec_emb_layer, decoder_inputs, decoder_lstm, attn_layer, decoder_dense)
    
    for i in range(4,6):
        print("Review:",seq2text(X_train[i], reverse_source_word_index))
        print("Original summary:",seq2summary(y_train[i],target_word_index, reverse_target_word_index))
        print("Predicted summary:",decode_sequence(X_train[i].reshape(1,max_len_text),encoder_model, decoder_model, target_word_index, reverse_target_word_index))
        print("\n")
    
    
    
if __name__ == '__main__':
    # sys.exit(main())
    main(1)