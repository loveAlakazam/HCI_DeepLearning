import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
# directory
current_dir = os.path.dirname(os.path.abspath(__file__))


categories = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
              5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}


def unpickle(data_batch_file):
    import pickle
    with open(data_batch_file, 'rb') as fo:
        dic1 = pickle.load(fo, encoding='bytes')
        return np.array(dic1[b'batch_label']), np.array(dic1[b'labels']), \
            np.array(dic1[b'data']), np.array(dic1[b'filenames'])

batch_labels, labels, data, filenames = \
        unpickle('cifar-10-batches-py/data_batch_2')

print('shape of batch_labels =', batch_labels) # b'training batch i of 5'
print('shape of labels =', labels.shape) # (10000,)
print('shape of CIFAR data =', data.shape) # (10000, 3072)
print('shape of filenames =', filenames.shape) # (10000,)

def CIFAR_image_to_gray_image(cifar_image):
    r_pane = cifar_image[0: 32*32] # R
    g_pane = cifar_image[32*32: 32*32*2] # G
    b_pane = cifar_image[32*32*2: 32*32*3] # B
    
    gray_image = r_pane * 0.2989 + g_pane * 0.5870 + b_pane * 0.1140
    return gray_image

def plot_gray_image(gray_image, label_val):    
    image = gray_image.reshape(32, 32)
    title = categories[np.argmax([label_val])] + ':' + str(label_val)

    fig = plt.figure(figsize=(3, 2))  # 가로 세로 길이(인치)    
    subplot = fig.add_subplot(1, 1, 1)
    subplot.set_xticks([])
    subplot.set_yticks([])    
    subplot.set_title(title)
    
    subplot.imshow(image, cmap=plt.cm.gray, interpolation='nearest')

def plot_CIFAR_10_image(index):
    img = data[index]
    R = img[0:1024].reshape(32,32)/255.0
    G = img[1024:2048].reshape(32,32)/255.0
    B = img[2048:].reshape(32,32)/255.0
    rgb_img = np.dstack((R,G,B))
    
    category = categories[labels[index]]
    file_name = str(filenames[index])
    
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.set_title(category + ', ' + file_name, fontsize =13)
    ax.imshow(rgb_img,interpolation='bicubic')

def to_one_hot_encoding_format(labels):
    r = []
    for i in labels:
        encoding = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # category 10개
        encoding[i] = 1
        r.append(encoding)

    one_hot_encoding_labels =np.array(r)   
    print('shape of one_hot_encoding_lables =', one_hot_encoding_labels.shape)
    print('one_hot_encoding_lables[' + str(i) + '] = ', one_hot_encoding_labels[0])   
    return one_hot_encoding_labels
    
    
def CIFAR_data_to_gray_data(CIFAR_data, labels, pkl_name):
    gray_data = []
    for i in range(len(CIFAR_data)):
        gray_data.append(CIFAR_image_to_gray_image(CIFAR_data[i]))
    
    r = np.array(gray_data)
    print('shape of grey_data = ', r.shape)
    
    fp = open(current_dir+'/'+ pkl_name, 'wb')
    pickle.dump(r, fp) # r의 shape은 (10000, 1024)
    one_hot_encoding_format_labels = to_one_hot_encoding_format(labels)
    pickle.dump(one_hot_encoding_format_labels, fp)
    fp.close()
    
    return r

def load_cifar_gray_data_and_one_labels(pkl_name):
    fp = open(current_dir+'/'+pkl_name, 'rb')
    grey_data = pickle.load(fp)
    grey_ohe_labels = pickle.load(fp)
    print('shape of gray data = ', grey_data.shape)
    print('shape of gray one-hot-encoding labels = ', 
          grey_ohe_labels.shape)
    #grey_data(10000,1024), grey_ohe_labels(10000,10)
    return grey_data, grey_ohe_labels 

# (10000, 1024)의 gray image 객체와 (10000,)의 1-hot-encoding 라벨
if __name__ == '__main__':
    CIFAR_data_to_gray_data(data, labels,pkl_name='cifar_grey_data_and_labels.pkl')

    for i in range(4):
        plot_CIFAR_10_image(i)

    gray_data, gray_labels = load_cifar_gray_data_and_ohe_labels('cifar_grey_data_and_labels.pkl')

    for i in range(4):
        plot_gray_image(gray_data[i], gray_labels[i])