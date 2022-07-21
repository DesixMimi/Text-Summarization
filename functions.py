from imports import *

from configparser import ConfigParser
config = ConfigParser()
config.read('config_var.ini')
max_len_text = config.getint('text_len', 'max_len_text')
max_len_summary = config.getint('text_len', 'max_len_summary')
EMBEDDING_DIM = config.getint('dimensions', 'EMBEDDING_DIM')

def load_dataset_tf():
    ds, info = tfds.load("cnn_dailymail", split="train", try_gcs=True, with_info=True)
    # Load the tfrecord and create the tf.data.Dataset
    assert isinstance(ds, tf.data.Dataset)
    print("done")
    return ds, info


def print_ds_info(ds,info, i=3):
    print(info.splits['train'].num_examples,"\n")
    print(info.splits['test'].num_examples,"\n")
    print(info.features,"\n")
    print(info.features['article'],"\n")
    
    for example in ds.take(i):
        # print(list(example.keys()))
        article, highlights = example["article"], example["highlights"]
        print(article)
    print("Example's type",type(example))


def bytes_to_str(df):
    # Changing the dta from bytes to string
    # print(type(df['text'][0]))
    # print(type(df['y'][0]))

    df['text'] = df['text'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    df['y'] = df['y'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    # print(type(df['text'][0]))
    # print(type(df['y'][0]))
    
    return df


def create_df_from_tf(ds_size=20000):
    ds, info = load_dataset_tf()
    # print_ds_info(ds,info,3)
    tf_df = tfds.as_dataframe(ds.take(ds_size), info)
    df = pd.DataFrame(tf_df).rename(columns={"article":"text", 
                                             "highlights":"y"})[["text","y"]]
    # When uploading data from tf it'll be in bytes
    df = bytes_to_str(df)
    
    return df


def df_to_csv(df):
    df.to_csv("./data/cnn_dailymail.csv")
    print("The df has been saved in the data folder")


def csv_to_df():
    df = pd.read_csv('./data/cnn_dailymail.csv')
    
    return df


## cleaning function
def utils_preprocess_text(txt, punkt=True, lower=True, slang=True, lst_stopwords=None, stemm=False, lemm=True):
    ### separate sentences with '. '
    txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))
    ### remove punctuations and characters
    txt = re.sub(r'[^\w\s]', '', txt) if punkt is True else txt
    ### strip
    txt = " ".join([word.strip() for word in txt.split()])
    ### lowercase
    txt = txt.lower() if lower is True else txt
    ### slang
    txt = contractions.fix(txt) if slang is True else txt   
    ### tokenize (convert from string to list)
    lst_txt = txt.split()
    ### stemming (remove -ing, -ly, ...)
    if stemm is True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_txt = [ps.stem(word) for word in lst_txt]
    ### lemmatization (convert the word into root word)
    if lemm is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_txt = [lem.lemmatize(word) for word in lst_txt]
    ### remove Stopwords
    if lst_stopwords is not None:
        lst_txt = [word for word in lst_txt if word not in 
                   lst_stopwords]
    ### back to string
    txt = " ".join(lst_txt)
    return txt

def word_count_plot(df):
    text_word_count = []
    summary_word_count = []

    # populate the lists with sentence lengths
    for i in df["text_clean"]:
        text_word_count.append(len(i.split()))

    for i in df["y_clean"]:
        summary_word_count.append(len(i.split()))

    length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})
    length_df.hist(bins = 30)
    # plt.show()
    length_df.describe()
    plt.savefig('./output/word_count_plot.png')
    

def print_clean_text(df):
    for i in range(3):
        print("Review:",df["text_clean"][i])
        print("Summary:",df["y_clean"][i])
        print("\n")
        

def truncate_text_length(df):
    cleaned_text =np.array(df['text_clean'])
    cleaned_summary=np.array(df['y_clean'])

    short_text=[]
    short_summary=[]

    for i in range(len(cleaned_text)):
        if(len(cleaned_summary[i].split())<=max_len_summary and len(cleaned_text[i].split())<=max_len_text):
            short_text.append(cleaned_text[i])
            short_summary.append(cleaned_summary[i])
            
    df_short=pd.DataFrame({'text_clean':short_text,'y_clean':short_summary})
    
    return df_short

def X_y_tokenization(X_y_train, X_y_test):
    # A tokenizer builds the vocabulary and converts a word sequence to an integer sequence.
    #prepare a tokenizer for news on training data
    X_y_tokenizer = Tokenizer()
    # fit_on_texts - updates internal vocabulary based on a list of texts.
    # In the case where texts contains lists, we assume each entry of the lists to be a token.
    X_y_tokenizer.fit_on_texts(list(X_y_train))

    #convert text sequences into corresponding integers from our word index that is inside our tokenizer
    # word index is a dictionary where the keys are the words found in the corpus and the values are just numbers assigned to each word
    X_y_train_seq = X_y_tokenizer.texts_to_sequences(X_y_train) 
    X_y_test_seq = X_y_tokenizer.texts_to_sequences(X_y_test)

    #padding zero upto maximum length
    X_y_train = pad_sequences(X_y_train_seq,  maxlen=max_len_text, padding='post', truncating = 'post') 
    X_y_test = pad_sequences(X_y_test_seq, maxlen=max_len_text, padding='post', truncating = 'post')

    X_y_voc_size = len(X_y_tokenizer.word_index) +1
    
    return X_y_train, X_y_test, X_y_voc_size, X_y_tokenizer
    

def glove_100d_dictionary():
    '''computing an index mapping words to known embeddings,
    by parsing the data dump of pre-trained embeddings:
    '''
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open('./data/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], dtype='float32')
        # Making a dictionary where each word is the key and the corresponding value is the 100d vector representing this word (from the trained model)
        embeddings_index[word] = vectors
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def map_word_to_glove(x_tokenizer, embeddings_index):
    # Creating embedding matrix
    word_index = x_tokenizer.word_index
    # Creating array of zeros
    embedding_matrix = np.zeros((len(word_index) + 1, 100)) # 100 because the glove we chose has 100d

    # For each word and it's index (from our corpus) we are going to assign the vector (from GloVe)
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix, word_index

def print_embd_explain(embedding_matrix,embeddings_index, word_index):
    # The position on the vectors will correspond to the position of words in our word_index (derived from our corpus)
    # Which means -> if we had the word "child" at index 0 in word_index, the vector in the embedding_matrix at index 0 will represent the word "child"
    print("The index (value) of the word child (key) is:", word_index["child"])
    print("\n")
    print("The vector from the pre-trained model for the word child is:", embeddings_index.get("child"))
    print("\n")
    print("This vector will be inside the embedding_matrix at position:", word_index["child"])
    print("\n")
    # Checking if the embedded matrix was created correctly
    print("Checking if the vectors match:", (embedding_matrix[20]== embeddings_index.get("child")).all())
    
    
