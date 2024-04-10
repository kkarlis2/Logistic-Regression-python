from os import walk,path
from collections import Counter


class Data:

    def __init__(self):
        self.url = self.link()
        self.mostCommonWordsToKeep =220  #Dwse poses apo tis pio synh8ismenes le3eis 8es na krathsoyme
        self.mostCommonWordsToDiscard=20  #Dwse poses apo tos pio synh8ismenes le3eis 8es na aporripsoyme

    def link(self):
        url= 'C:\\Users\\kosta\\Desktop\\aclImdb'  #Dwse to path poy brisketai to aclImdb ston ypologisti soy
        return url

    def read_train(self):    #diabase ti brisketai ston fakelo ekpaideyshs kai epestre3e to
        return self.url + '\\train'

    def read_test(self):    #diabase ti brisketai ston fakelo testarismatos kai epestre3e to
        return self.url + '\\test'

    def get01(self,data,typos): #synarthsh ypologismoy dianismatwn  
        if data == 'test':
            url = self.read_test()+'\\'
        elif data == 'train':
            url=self.read_train()+'\\'
        else:
            return

        if typos!='neg' and typos != 'pos':
            return

        values = []
        file=[]

        for(dirpath,folders,files) in walk(url + typos):
            file.extend(files)
        for i in file:
            ls=[]
            dianisma=[]
            ls.extend(self.read_file(url+typos+'\\'+i))
            dc=open(self.url+'\\'+'dictionary.txt',encoding="utf8")
            line=dc.readline().strip()
            while line:
                if line in ls:
                    dianisma.append(1)
                else:
                    dianisma.append(0)
                line=dc.readline().strip()
            if typos=='neg':
                dianisma.append(0)
            else:
                dianisma.append(1)
            values.append(dianisma)
        dc.close()
        return values


    def createDictionary(self):  #dhmiourgia le3ilogiou me bash ta dedomena poy dw8hkan sthn __init__ me tis pio syxnes emfanizomenes le3eis
        if not path.exists(self.url+'\\'+'dictionary.txt'):
            wPos = ['neg', 'pos']
            words = []
            for i in wPos:
                for(dirpath, folders, files) in walk(self.read_train()+'\\'+i):
                    files.extend(files)
                for j in files:
                    words.extend(self.read_file(self.read_train()+'\\'+i+'\\'+j))
            wordsCount = (wPos for wPos in words)
            cw = Counter(wordsCount)
            cw = cw.most_common(self.mostCommonWordsToKeep)[self.mostCommonWordsToDiscard:]
            ww = [wPos[0] for wPos in cw]
            with open(self.url+'\\'+'dictionary.txt', "w", encoding="utf-8") as file:
                    for i in ww:
                        if not(i.startswith('/') or i.startswith('<') or i.startswith('>') or i.startswith('-') ):
                                file.write(i.lower()+"\n")
            file.close()



    def read_file(self,url): #synarthsh anagnwshs twn files
        ls=[]
        file=open(url,"r",encoding="utf-8")
        line = file.readline().strip()
        while line:
            ls.extend(line.split())
            line=file.readline().strip()
        file.close()
        return ls


Data().createDictionary() #dhmioyrghse to le3ilogio
print("Dictionary created!")
