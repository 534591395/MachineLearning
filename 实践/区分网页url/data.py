import pandas as  pd
import tensorflow as tf


#https://www.jianshu.com/p/ff8e5f4635cc
#https://blog.csdn.net/john_xyz/article/details/54706807
#https://blog.csdn.net/lk7688535/article/details/52798735
TRAIN_URL = "http://10.240.192.60:3100/tests/training.csv"
TEST_URL = "http://10.240.192.60:3100/tests/test.csv"

CSV_COLUMN_NAMES = ['url', 'class']

def download():
    train_path = tf.keras.utils.get_file('training', TRAIN_URL)
    test_path = tf.keras.utils.get_file('test', TEST_URL)
    return train_path, test_path

def load_data(y_name="class"):
    train_path, test_path = download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)
    
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
    #print(train_x, train_y)
    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(10000).repeat().batch(batch_size)
    return dataset

def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    
    return dataset

def padding_sentences(input_sentences, padding_token, padding_sentence_length=None):
    sentences = [getTokens(sentence) for sentence in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max(
            [len(sentence) for sentence in sentences])
    all_vector=[]
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
        all_vector.append(sentence)
    return (all_vector, max_sentence_length)

"""data帮助"""
def getTokens(input):
    allTokens = []
    token = str(input)
    tokens = []
    #token = token.replace('?', '/')
    tokens = token.split('?')
    allTokens = allTokens + tokens
    return allTokens    

if __name__ == '__main__':
   #load_data()
   test = "rapiseebrains.com/?a=401336&c=cpc&s=050217"
   ans = getTokens(test)
   print(ans, len(ans))