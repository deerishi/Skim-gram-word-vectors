
# coding: utf-8

# In[1]:

# imports
# %pylab osx
import tensorflow as tf
import numpy as np


from collections import deque,Counter
# Some additional libraries which we'll use just
# to produce some visualizations of our training
import re
import random
import pandas as pd
# Bit of formatting because I don't like the default inline code style:


# In[45]:

github=pd.read_csv('15kfilteredComments.csv')
#print(github)
data=list(github['body'])
#print(data[:10])
vocab=[]
[vocab.extend(l.split(' ')) for l in data]
data=[w for w in vocab  if len(w)>0]
print(data[:10])
vocab=list(set(data))
print(len(vocab))
#common=Counter(data).most_common(50000)
#print(common)
parentFolder='github_word_vectors'





# In[ ]:

encoder=dict(zip(vocab,range(len(vocab))))
decoder=dict(zip(encoder.values(),encoder.keys()))
#print(encoder.values())
data_index=0
encoded_data=[encoder[w] for w in data]
#np.save('encoder',encoder)
#np.save('decoder',decoder)
#print(encoded_data)


# In[46]:

def buildSecondDataset(dataRaw,vocab_size=80000):
    count=[['UNK#',-1]]
    count.extend(Counter(data).most_common(vocab_size-1))
    encoder=dict()
    for word,_ in count:
        #print('word is ',word)
        encoder[word]=len(encoder)
    index=0
    for i in range(len(data)):
        word=data[i]
        if word not in encoder:
            data[i]='UNK#'
    count=[]
    count.extend(Counter(data).most_common(vocab_size-1))        
    #print(count)
    print(len(list(set(data))))
    decoder=dict(zip(encoder.values(),encoder.keys()))
    np.save(parentFolder+'/encoderGithub'+'_'+str(len(encoded_data)),encoder)
    np.save(parentFolder+'/decoderGithub'+'_'+str(len(encoded_data)),decoder)
    return encoder,decoder
encoder,decoder=buildSecondDataset(data)
data_index=0
encoded_data=[encoder[w] for w in data]
#print('decoder is ',decoder)
vocab=list(encoder.values())
print('vocab is ',len(vocab))


# In[40]:

def generate_batch(batch_size=8,num_skips=4,skip_window=2):
    global data_index
    batch=[]
    old_value=data_index
    labels=[]
    span=2*skip_window+1

    buffer=deque(maxlen=span)
    for _ in range(span):
        buffer.append(encoded_data[data_index])
        data_index=(data_index+1)%len(encoded_data)
    for i in range(batch_size // num_skips):
        for j in range(num_skips):
            targets=[skip_window]
            batch.append(buffer[skip_window])
            target=skip_window
            while target in targets:
                target=random.randint(0,span-1)
            labels.append(buffer[target])
        buffer.append(encoded_data[data_index])
        data_index=(data_index+1)%len(encoded_data)
    batch=np.asarray(batch)
    labels=np.asarray(labels)
    labels=labels.reshape(-1,1)
    data_index=(old_value+skip_window)%len(encoded_data)
    return batch,labels
            
    


# In[42]:

print(data[0:20])
global data_index
data_index=0
b,l=generate_batch()
#print(b,l)
print('batch is ',[decoder[i] for i in b],' and labels are ',[decoder[j] for j in l[:,0]])
b,l=generate_batch()
#print(b,l)
print('batch is ',[decoder[i] for i in b],' and labels are ',[decoder[j] for j in l[:,0]])


# In[47]:

from tensorflow.python.framework import ops
ops.reset_default_graph()

embeddings_size=100
num_sampled=100
X=tf.placeholder(name='input',dtype=tf.int32)
Y=tf.placeholder(name='output',dtype=tf.int32)


weights=tf.get_variable(name='weights',shape=[len(vocab),embeddings_size],
                        initializer=tf.random_uniform_initializer(-1,1))
biases=tf.get_variable(name='biases',shape=[len(vocab)],
                        initializer=tf.constant_initializer(0.0))

embeddings=tf.get_variable(name='embeddings',shape=[len(vocab),embeddings_size],
                        initializer=tf.random_uniform_initializer(-1,1))

embed=tf.nn.embedding_lookup(embeddings,X)

cost=tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=weights,biases=biases,num_sampled=100,
                                               num_classes=len(vocab),labels=Y,inputs=embed))

optimizer=tf.train.AdagradOptimizer(learning_rate=1.0).minimize(cost)

valid_size = 64 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


# In[49]:

with tf.Session() as sess:
    batch_size=8
    n_iterations=(len(encoded_data)//batch_size )*2
    global data_index
    data_index=0
    tcost=0
    sess.run(tf.global_variables_initializer())
    for i in range(1,n_iterations):
        b,l=generate_batch(batch_size=batch_size,num_skips=4,skip_window=2)
        #print('b is ',b,' l is ',l)
        _,c=sess.run([optimizer,cost],feed_dict={X:b,Y:l})
        tcost+=c
        if i%20000==0:
            print('Average cost is ',tcost/20000)
            tcost=0.0
        if i % 20000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = decoder[valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = decoder[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
            print('\n\n')
    final_embeddings = normalized_embeddings.eval()
    np.save(parentFolder+'/github_final_embeddings'+str(len(encoded_data)),final_embeddings)
print('Completed')

# In[ ]:


