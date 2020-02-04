import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import jieba
import itertools
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

def JeibaCutWords(input_df):
    cols = ["Description"]  # 決定要選取那些列，當作文字來源(可複選)
    corpus_cutted = list([])  # 儲存斷詞後的句子
    corpus_class = list([]) # 句子的分類
    corpus_id = list([]) # 句子的編號

    # 設定字典
    jieba.set_dictionary('./Jeiba/dict.txt.big')
    jieba.load_userdict('./Jeiba/my.dict.txt')  # 載入自訂字典

    # 設定停用詞
    with open(r'./Jeiba/stop_words.txt', 'r', encoding='utf8') as f:  
        stops = f.read().split('\n') 

    for row_index, row in enumerate(input_df.index):
        #print("[%d]" % (row))
        temp_str = str()
        one_line_cutted = list([])
        for col_index, column in enumerate(cols):
            sentence = input_df.loc[row, column]  # 選取一列 
            temp_str = temp_str + sentence + "," # 欄位資料合併
        #print("合併後句子: %s" % (temp_str))
        
        #
        # 結巴斷詞
        #
        for word in jieba.cut(temp_str, cut_all=True):
            if word not in stops:
                one_line_cutted.append(word)

        #print(one_line_cutted)
        corpus_cutted.append(one_line_cutted)  # 斷詞後的每篇廣告
        corpus_class.append(input_df.loc[row, 'Class'])
        corpus_id.append(input_df.loc[row, 'ID'])
    
    # 收集完的資料存成dataframe
    temp_data = {"id": corpus_id,
                 "sentence": corpus_cutted,
                 "class": corpus_class
                }
    temp_df = pd.DataFrame(temp_data)
    return temp_df


# 若是檢查到違法的關鍵字，直接標註為違法廣告，並記錄在keyword_flag
def AppendKeywordCheck(input_df):
    keyword_flag = np.zeros((input_df.shape[0],), dtype=np.int)  # 建立一個key word check陣列

    #
    # 逐一檢查是否有違規字眼，若有找到，直接標註其class為違法廣告
    #
    for index, row in input_df.iterrows():  # 每一則廣告
        illegal_word_list = list([])
        for token in enumerate(row['sentence']):  # 每一個詞
            f = open('./Jeiba/illegal.keywords.txt', 'r', encoding='UTF-8')  # 違規字詞的字典
            for illegal_keyword in f.readlines():
                illegal_keyword = illegal_keyword.replace('\n', '')
                #
                # 比較該詞是否為違法字詞
                #
                if token[1] == illegal_keyword:
                    illegal_word_list.append(token[1]) 
            f.close()

        # 列印出含有違規字詞的廣告
        if len(illegal_word_list) != 0:
            #print("---------------------------------------------------------------------------")
            #print("ID:[{0}], class:({1}) 內容:{2}".format(index, row['class'], row['sentence']))
            #print("違法字詞: {0}".format(illegal_word_list))

            # 設定該class為違法廣告
            keyword_flag[index] = 1  # VIOLATE_CLASS
            
    return keyword_flag, illegal_word_list


def PlotWordCloud(words_source):
    terms_dict = Counter(words_source)  # 把字詞統計資換轉換成dictionary
    #print(terms_dict.items())

    font = r'msjh.ttf'
    #想要文字雲出現的圖示
    #mask = np.array(Image.open(r"mayday_mask.png"))

    #其他參數請自行參考wordcloud
    my_wordcloud = WordCloud(background_color="white",font_path=font,collocations=False, width=2400, height=2400, margin=1)  
    my_wordcloud.generate_from_frequencies(frequencies=Counter(words_source))

    #產生圖片
    plt.figure( figsize=(8,4), facecolor='k')
    plt.imshow(my_wordcloud,interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    #顯示用
    plt.show()
    

def ShowWordCloud(input_df):
    legal_terms = list([])  # 儲存合法廣告的單詞
    violate_terms = list([])  # 儲存違法廣告的單詞
    
    for index, row in input_df.iterrows():
        #print(row['sentence'])
        if row['class'] == 0:  # 
            for word in row['sentence']:
                #print(word)
                legal_terms.append(word)  # 收集合法廣告詞
        if row['class'] == 1:  # 
            for word in row['sentence']:
                violate_terms.append(word)  # 收集違法廣告詞
    print("合法廣告文字雲:")
    PlotWordCloud(legal_terms)
    print("違法廣告文字雲:")
    PlotWordCloud(violate_terms)
    

#%matplotlib inline
#from matplotlib import rcParams


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure(figsize=(8,4))
    #rcParams['figure.figsize'] = (8.0, 4.0)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()