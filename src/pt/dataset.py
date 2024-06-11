from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
import torch


class TaggerRewriterDataset(Dataset):

    def __init__(self, df, tokenizer, valid=False):
        self.a = df['a'].values.tolist() # 用户输入
        self.b = df['b'].values.tolist() # 系统回复
        self.is_valid = valid
        self.current = df['current'].values.tolist() # 待改写语句
        self.label = df['label'].values.tolist() # 理想改写结果
        self._tokenizer = tokenizer
        self.ori_sentence = []
        self.sentence = []
        self.token_type = []
        self.pointer = []
        self.context_len = []
        self.valid_index = []
        self.valid_label = []
        self.label_type = []
        self.generate_label()

    def tokenize_chinese(self,sen):
        temp = []
        for word in sen:
            if word in self._tokenizer.vocab:
                temp.append(word)
            else:
                temp.append("[UNK]")
        return temp

    def generate_label(self):
        """改写数据生成
        1. 改写这里作者分了两中常见情景。
        - 30%包含指代词。我喜欢他 -> 我喜欢张艺兴
        - 50%会有信息省略。快告诉我 -> 快告诉我现在几点
        Args:

        Returns:
            start: 关键信息start（关键主语等信息）
            end: 关键信息end（关键主语等信息）
            insert: 补全位置
            start_ner: 指代start位置
            start_end: 指代end位置
        """
        # 全部采用指针抽取
        # 根据改写的数据对原始数据进行标注
        # 去哪里    长城北路公园    在什么地方     长城北路公园在什么地方
        # 确实江西炒粉要用瓦罐汤 特产 没错是我老家的特产 没错江西炒粉是我老家的特产
        # 为什么讨厌张艺兴       我喜欢张艺兴 很可爱啊       我也喜欢他     我也喜欢张艺兴

        # start,end,insert,start_ner,end_ner
        drop_item = 0
        for i in range(len(self.a)):
            # 生成随机数决定样本要不要改写，否则把label作为current
            n = random.random()
            start, end, insert_pos, start_ner, end_ner = 0,0,0,0,0
            new_token_type = []
            # 上下文: 用户输入 + 系统回复
            context_new_input = ["[CLS]"]+self.tokenize_chinese(self.a[i])+["[SEP]"]+self.tokenize_chinese(self.b[i])+["[SEP]"]
            new_token_type.extend([0]*len(context_new_input))
            if n >= 0.3:
                utterance_token = self.tokenize_chinese(self.current[i])+["SEP"]
            else:
                # 直接拿label构建数据，就不需要改写了，相当于构造了负样本
                utterance_token = self.tokenize_chinese(self.label[i])+["SEP"]
            # context_new_input: 用户输入 + 系统回复
            # new_input: 上下文 + 当前待改写的文本
            new_input = context_new_input + utterance_token

            # 上下文token_type为0，当前token_type为1
            new_token_type.extend([1]*len(utterance_token))

            # 改写或者作为验证集时不对关键信息进行抽取
            if self.is_valid or n<0.3 and False:
                # 改写的负样本
                if self.is_valid:
                    _label = [None] * 5
                else:
                    _label = [0]*5
                self.pointer.append(_label)
                self.sentence.append(self._tokenizer.convert_tokens_to_ids(new_input))
                self.token_type.append(new_token_type)
                self.context_len.append(context_new_input)
                self.ori_sentence.append([',', self.a[i], ',' + self.b[i], ',', self.current[i], ','])
                self.valid_index.append(i)
                self.valid_label.append(self.label[i])
                self.label_type.append([0, 0])
                continue
            # 获取四个指针信息
            insert = True
            # 如果原始语句所有词汇都在改写中，则改写为插入新语句（而不是替换现有的一些词汇）
            for word in self.current[i]:
                if word not in self.label[i]:
                    insert = False
            # -----寻找增加的信息------------------
            '''
            - 寻找关键信息起点
            1. 如果前面的都相等，那么起点位置就是len(current), 例如current=我也想去，label=我也想去听演唱会
            2. 如果中间出现不一致的字符，那么中间的不想等的位置则为起点，例如current=周杰伦的比较难抢，label=周杰伦的演唱会比较难抢
            '''
            text_start, text_end = 0, 0
            for j in range(len(self.label[i])):
                if j >= len(self.current[i]):
                    text_start = j
                    break
                if self.current[i][j] == self.label[i][j]:
                    continue
                else:
                    text_start = j
                    break
            '''
            - 寻找关键信息的终点位置
            1. 从后往前看，如果相等，则跳过。
            2. 如果不想等，则找到了结束位置。例如current=周杰伦的【end】比较难抢，label=周杰伦的演唱会【end】比较难抢
            3. 如果一致相等，则当前节点就为end位置。例如current=【end】我也想去，label=演唱会【end】我也想去
            '''
            for j in range(len(self.label[i])):
                if j >= len(self.current[i]):
                    text_end = j
                    break
                if self.current[i][::-1][j] == self.label[i][::-1][j]:
                    continue
                else:
                    text_end = j
                    break
            # 这里抽取出来的text则为改写后label中补充的text
            text = self.label[i][text_start:(len(self.label[i]) - text_end)]
            # 获取插入文本及位置：找到补充的text的来源
            if text in self.a[i]:
                start = self.a[i].index(text) + 1
                end = start + len(text) - 1
            elif text in self.b[i]:
                start = self.b[i].index(text) + len(self.a[i]) + 2
                end = start + len(text) - 1
            else:
                # 如果没有，则跳过该样本
                drop_item += 1
                continue
            if insert:
                self.label_type.append(0)
                # 去哪里    长城北路公园    在什么地方     长城北路公园在什么地方
                insert_pos = len(self.current[i])-text_end + len(context_new_input) # 为啥 + len(context_new_input)?
            else:
                # 指代
                # 为什么讨厌张艺兴       我喜欢张艺兴 很可爱啊       我也喜欢他     我也喜欢张艺兴
                # 和前面同样的方法找到指代词的起点终点位置
                coref_start, coref_end = 0, 0
                for j in range(len(self.current[i])):
                    if self.current[i][j] == self.label[i][j]:
                        continue
                    else:
                        coref_start = j
                        break
                for j in range(len(self.current[i])):
                    if self.current[i][::-1][j] == self.label[i][::-1][j]:
                        continue
                    else:
                        coref_end = j
                        break
                self.label_type.append(1)
                start_ner = coref_start+len(context_new_input)
                end_ner = len(self.current[i])-coref_end+len(context_new_input)-1
            # print(self.a[i],self.b[i],self.current[i],self.label[i], text)
            # print(start,end,insert_pos,start_ner,end_ner)
            if self.is_valid:
                self.pointer.append(_label)
            else:
                self.pointer.append([start,end,insert_pos,start_ner,end_ner]) # 关键信息起点，终点，插入位置，指代消解起点，终点
            self.sentence.append(self._tokenizer.convert_tokens_to_ids(new_input))  # new_input: 上下文 + 当前待改写的文本
            self.token_type.append(new_token_type)  # 上下文token_type为0，当前token_type为1
            self.context_len.append(context_new_input) # 上下文: 用户输入 + 系统回复
            self.ori_sentence.append(','+self.a[i]+','+self.b[i]+','+self.current[i]+',') # 原始句子
            self.valid_label.append(self.label[i]) # 理想的改写结果
            self.valid_index.append(i) # index
        print('数据总数 ', len(self.sentence), '丢弃样本数目 ', drop_item)
        print('信息插入', self.label_type.count(0))
        print('指代消歧义', self.label_type.count(1))


    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        return  self.ori_sentence[idx],\
                torch.LongTensor(self.sentence[idx]),  \
                torch.LongTensor(self.token_type[idx]),\
                self.pointer[idx][0],\
                self.pointer[idx][1],\
                self.pointer[idx][2],\
                self.pointer[idx][3],\
                self.pointer[idx][4]


def tagger_collate_fn(batch):
    # start, end, insert_pos, start_ner, end_ner = 0,0,0,0,0
    ori_sen, token, token_type, start, end,insert_pos,start_ner,end_ner = zip(*batch)
    token = pad_sequence(token, batch_first=True)
    token_type = pad_sequence(token_type, batch_first=True, padding_value=1)
    if start[0] is not None:
        start = torch.tensor(start)
        end = torch.tensor(end)
        insert_pos = torch.tensor(insert_pos)
        start_ner = torch.tensor(start_ner)
        end_ner = torch.tensor(end_ner)
    return ori_sen, token, token_type, start, end, insert_pos, start_ner, end_ner