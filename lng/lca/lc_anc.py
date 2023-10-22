import string,re,sys,os,random
from math import sqrt,log
from .anc_all_count import wordlist
# from .snli_freq import wordlist
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag, sent_tokenize
lemmatizer = WordNetLemmatizer()
from nltk.corpus import wordnet
tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
        }

# Returns the keys of dictionary d sorted by their values
def sort_by_value(d):
    items=d.items()
    backitems=[ [v[1],v[0]] for v in items]
    backitems.sort()
    return [ backitems[i][1] for i in range(0,len(backitems))]

# NDW for first z words in a sample
def getndwfirstz(z,lemmalist):
    ndwfirstztype={}
    for lemma in lemmalist[:z]:
        ndwfirstztype[lemma]=1
    return len(ndwfirstztype.keys())

# NDW expected random z words, 10 trials
def getndwerz(z,lemmalist):
    ndwerz=0
    for i in range(10):
        ndwerztype={}
        erzlemmalist=random.sample(lemmalist,z)
        for lemma in erzlemmalist:
            ndwerztype[lemma]=1
        ndwerz+=len(ndwerztype.keys())
    return ndwerz/10.0

# NDW expected random sequences of z words, 10 trials
def getndwesz(z,lemmalist):
    ndwesz=0
    for i in range(10):
        ndwesztype={}
        startword=random.randint(0,len(lemmalist)-z)
        eszlemmalist=lemmalist[startword:startword+z]
        for lemma in eszlemmalist:
            ndwesztype[lemma]=1
        ndwesz+=len(ndwesztype.keys())
    return ndwesz/10.0

# MSTTR
def getmsttr(z,lemmalist):
    samples=0
    msttr=0.0
    while len(lemmalist)>=z:
        samples+=1
        msttrtype={}
        for lemma in lemmalist[:z]:
            msttrtype[lemma]=1
        msttr+=len(msttrtype.keys())/float(z)
        lemmalist=lemmalist[z:]    
    return msttr/samples

def isLetterNumber(character):
    if character in string.printable and not character in string.punctuation:
        return 1
    return 0

def isSentence(line):
    for character in line:
        if isLetterNumber(character):
            return 1
    return 0

# reads information from bnc wordlist
adjdict={}
verbdict={}
noundict={}
worddict={}
wordlist = wordlist.split('\n')
# wordlistfile=open("anc_all_count.txt","r")
# wordlist=wordlistfile.readlines()
# wordlistfile.close()
for word in wordlist:
    wordinfo=word.strip()
    if not wordinfo or "Total words" in wordinfo:
        continue
    infolist=wordinfo.split()
    lemma=infolist[1]
    pos=infolist[2]
    frequency=int(infolist[3])
    worddict[lemma]=worddict.get(lemma,0)+frequency
    if pos[0]=="J":
        adjdict[lemma]=adjdict.get(lemma,0)+frequency
    elif pos[0]=="V":
        verbdict[lemma]=verbdict.get(lemma,0)+frequency
    elif pos[0]=="N":
        noundict[lemma]=noundict.get(lemma,0)+frequency
wordranks=sort_by_value(worddict)
verbranks=sort_by_value(verbdict)


# standard can be computed as the 20-th percentile of word counts:
#    lengths = [len(word_tokenize(x)) for x in data['train']['text']]
#    standard = int(np.percentile(lengths, 20))

def lca(input_text, standard = 8):
    lemlines = sent_tokenize(input_text)
    for i in range(len(lemlines)):
        morph = pos_tag(word_tokenize(lemlines[i]))
        lemlines[i] = " ".join(["{}_{}".format(
            lemmatizer.lemmatize(word, tag_dict[tag[0]] if tag[0] in tag_dict else 'n'),
            tag)
            for word, tag in morph])

# process input file
    wordtypes={}
    wordtokens=0
    swordtypes={}
    swordtokens=0
    lextypes={}
    lextokens=0
    slextypes={}
    slextokens=0
    verbtypes={}
    verbtokens=0
    sverbtypes={}
    adjtypes={}
    adjtokens=0
    advtypes={}
    advtokens=0
    nountypes={}
    nountokens=0
    lemmaposlist=[]
    lemmalist=[]

    for lemline in lemlines:
        lemline=lemline.strip()
        lemline=lemline.lower()
        if not isSentence(lemline):
            continue
        lemmas=lemline.split()
        for lemma in lemmas:
            word=lemma.split("_")[0]
            pos=lemma.split("_")[-1]
            if (not pos in string.punctuation) and pos!="sent" and pos!="sym":
                lemmaposlist.append(lemma)
                lemmalist.append(word)  
                wordtokens+=1
                wordtypes[word]=1
                if (not word in wordranks[-2000:]) and pos!="cd":
                    swordtypes[word]=1
                    swordtokens+=1
                if pos[0]=="n":
                    lextypes[word]=1
                    nountypes[word]=1
                    lextokens+=1
                    nountokens+=1
                    if not word in wordranks[-2000:]:
                        slextypes[word]=1
                        slextokens+=1
                elif pos[0]=="j":
                    lextypes[word]=1
                    adjtypes[word]=1
                    lextokens+=1
                    adjtokens+=1
                    if not word in wordranks[-2000:]:
                        slextypes[word]=1
                        slextokens+=1
                elif pos[0]=="r" and (word in adjdict or (word[-2:]=="ly" and word[:-2] in adjdict)):
                    lextypes[word]=1
                    advtypes[word]=1
                    lextokens+=1
                    advtokens+=1
                    if not word in wordranks[-2000:]:
                        slextypes[word]=1
                        slextokens+=1
                elif pos[0]=="v" and not word in ["be","have"]:
                    verbtypes[word]=1
                    verbtokens+=1
                    lextypes[word]=1
                    lextokens+=1
                    if not word in wordranks[-2000:]:
                        sverbtypes[word]=1
                        slextypes[word]=1
                        slextokens+=1

# 1. lexical density
    ld=float(lextokens)/max(wordtokens, 1)

# 2. lexical sophistication
# 2.1 lexical sophistication
    ls1=slextokens/max(1,float(lextokens))
    ls2=len(swordtypes.keys())/max(float(len(wordtypes.keys())), 1)

# 2.2 verb sophistication
    if verbtokens == 0:
        vs1 = 0
        vs2 = 0
        cvs1 = 0
    else:
        vs1=len(sverbtypes.keys())/float(verbtokens)
        vs2=(len(sverbtypes.keys())*len(sverbtypes.keys()))/float(verbtokens)
        cvs1=len(sverbtypes.keys())/sqrt(2*verbtokens)

# 3 lexical diversity or variation

# 3.1 NDW, may adjust the values of "standard"
    ndw=ndwz=ndwerz=ndwesz=len(wordtypes.keys())
    if len(lemmalist)>=standard:
        ndwz=getndwfirstz(standard,lemmalist)
        ndwerz=getndwerz(standard,lemmalist)
        ndwesz=getndwesz(standard,lemmalist)

# 3.2 TTR
    msttr=ttr=len(wordtypes.keys())/max(float(wordtokens), 1)
    if len(lemmalist)>=standard:
        msttr=getmsttr(standard,lemmalist)
    cttr=len(wordtypes.keys())/max(sqrt(2*wordtokens), 1)
    rttr=len(wordtypes.keys())/max(sqrt(wordtokens), 1)
    logttr=log(len(wordtypes.keys()))/log(wordtokens) if wordtokens > 1 else 0
    if wordtokens == len(wordtypes.keys()):
        uber = 0
    else:
        uber=(log(wordtokens,10)*log(wordtokens,10))/log(wordtokens/float(len(wordtypes.keys())),10)

# 3.3 verb diversity
    if verbtokens == 0:
        vv1, svv1, cvv1 = 0, 0, 0
    else:
        vv1=len(verbtypes.keys())/float(verbtokens)
        svv1=len(verbtypes.keys())*len(verbtypes.keys())/float(verbtokens)
        cvv1=len(verbtypes.keys())/sqrt(2*verbtokens)

# 3.4 lexical diversity
    lv=len(lextypes.keys())/max(1,float(lextokens))
    vv2=len(verbtypes.keys())/max(1,float(lextokens))
    nv=len(nountypes.keys())/max(1,float(nountokens))
    adjv=len(adjtypes.keys())/max(1,float(lextokens))
    advv=len(advtypes.keys())/max(1,float(lextokens))
    modv=(len(advtypes.keys())+len(adjtypes.keys()))/max(1,float(lextokens))


    output = [len(wordtypes.keys()), len(swordtypes.keys()), len(lextypes.keys()),
            len(slextypes.keys()),
            wordtokens, swordtokens, lextokens, slextokens,
            ld, ls1, ls2, vs1, vs2, cvs1, ndw, ndwz, ndwerz, ndwesz, ttr, msttr, cttr,
            rttr, logttr, uber, lv, vv1, svv1, cvv1, vv2, nv, adjv, advv, modv]
    return output
