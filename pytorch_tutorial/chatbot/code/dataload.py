import random
import unicodedata
import string
import re
import torch
import pickle
def readLines(file_name):
    with open(file_name,'rb') as f:
        lines = f.readlines()
        return lines

def lineProcess(file_name,fields):
    lines = readLines(file_name)
    pattern = " +++$+++ "
    sentence = [str(line, encoding='iso-8859-1').split(pattern)
                for line in lines]
    lines = {}
    for line in sentence:
        lineObj = {}
        for i,field in enumerate(fields):
            lineObj[field] = line[i]
        lines[lineObj['lineID']] = lineObj
    return lines

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def loadConversations(file_name,fields):
    conversations = []
    with open(file_name,'r',encoding='iso-8859-1') as f:
        for line in f:
            content = line.split(' +++$+++ ')
            #print(content)
            convObj = {}
            for i,filed in enumerate(content):
                convObj[fields[i]] = content[i]
            conversations.append(convObj)

    return conversations


def extractPairs(lines,conversations):
    conversation_pairs = []
    for con in conversations:
        utterance_ids = eval(con['utteranceIDs'])
        for i in range(len(utterance_ids)-1):
            input_line = lines[utterance_ids[i]]['text'].rstrip('\n')
            output_line = lines[utterance_ids[i+1]]['text'].rstrip('\n')
            len1 = len(input_line.split(' '))
            len2 = len(output_line.split(' '))
            input_line = normalizeString(input_line)
            output_line = normalizeString(output_line)
            if len1 <= MAX_LENGTH and len2 <= MAX_LENGTH:
                conversation_pairs.append([input_line,output_line])

    return conversation_pairs


MAX_LENGTH = 10
file_name = '../data/cornell movie-dialogs corpus/movie_lines.txt'
LINE_FIELDS = ['lineID','characterID','movieID','character','text']
lines = lineProcess(file_name,LINE_FIELDS)
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
conversation_file = '../data/cornell movie-dialogs corpus/movie_conversations.txt'
conversations = loadConversations(conversation_file,MOVIE_CONVERSATIONS_FIELDS)
conversation_pairs = extractPairs(lines,conversations)

pair = random.choice(conversation_pairs)

PAD_token = 0
SOS_token = 1
EOS_token = 2
#voc 的作用,统计
class VOC():
    def __init__(self):
        self.n_words = 3
        self.is_trimmed = False
        self.word2Count = {}
        self.word2Index = {}
        self.index2Word = {PAD_token:'PAD',SOS_token:'SOS',EOS_token:'EOS'}

    def addWord(self,word):
        if word not in self.word2Count:
            self.word2Count[word] = 1
            self.word2Index[word] = self.n_words
            self.index2Word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2Count[word] += 1
    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def trim(self,min_count):
        if self.is_trimmed:
            return
        else:
            self.word2Index = {}
            self.index2Word = {}
            self.n_words = 3
            for word,count in self.word2Count.items():
                if count >= min_count:
                    self.word2Index[word] = self.n_words
                    self.index2Word[self.n_words] = word
                    self.n_words += 1

def trimRareWord(conversation_pairs,voc):
    trimmed_conversations =  []
    for pair in conversation_pairs:
        keep_input = True
        keep_output = True
        for word in pair[0].split(' '):
            if word not in voc.word2Index:
                keep_input = False
                break
        for word in pair[1].split(' '):
            if word not in voc.word2Index:
                keep_output = False
                break
        if keep_output and keep_output:
            trimmed_conversations.append(pair)

    return trimmed_conversations
voc = VOC()
MIN_COUNT = 3
#construct the vocabulary
for pair in conversation_pairs:
    for p in pair:
        voc.addSentence(p)
voc.trim(MIN_COUNT)
trimed_pairs = trimRareWord(conversation_pairs,voc)
voc_file = "../data/voc.pkl"
with open(voc_file,'wb') as f:
    pickle.dump(voc,f)

def writePairsToFile(file_name,pairs):
    with open(file_name,'w',encoding='utf-8') as f:
        for p in pairs:
            conversation = '+$+'.join(p)
            f.write(conversation+'\n')
#writePairsToFile(file_write_path,trimed_pairs)
file_write_path = '../data/conversation_pais.txt'

