import re
file_name = '../data/cornell movie-dialogs corpus/movie_lines.txt'
def readLines(file_name):
    with open(file_name,'rb') as f:
        lines = f.readlines()
        return lines


pattern = " +++$+++ "

def lineProcess(file_name,fields):
    lines = readLines(file_name)
    sentence = [str(line, encoding='iso-8859-1').split(pattern)
                for line in lines]
    lines = {}
    for line in sentence:
        lineObj = {}
        for i,field in enumerate(fields):
            lineObj[field] = line[i]
        lines[lineObj['lineID']] = lineObj
    return lines

fields = ['lineID','characterID','movieID','character','text']
lines = lineProcess(file_name,fields)
