import shifterator as sh
from collections import Counter
from labMTsimple.storyLab import emotionFileReader, emotion, stopper, emotionV


labMT,labMTvector,labMTwordList = emotionFileReader(stopval=0.0,lang='english', returnVector=True)



# read in the two corpora
with open('data/one_grams_clean_first.txt') as f:
    words1 = f.readlines()
    
    
with open('data/one_grams_clean_second.txt') as f:
    words2 = f.readlines()


def prep_corpus(corpus1, corpus2):  # Returns allowable objects to use as arguments to Shifterator functions  
    for i in range(len(corpus1)):
        corpus1[i] = corpus1[i][:-1]
    
    for i in range(len(corpus2)):
        corpus2[i] = corpus2[i][:-1]

    lefthandcounter = Counter(corpus1)  
    righthandcounter = Counter(corpus2)  
    for key in lefthandcounter.keys():    
        if key not in righthandcounter.keys():      
            righthandcounter[key] = 0  
    for key in righthandcounter.keys():    
        if key not in lefthandcounter.keys():      
            lefthandcounter[key] = 0  
    
    return(dict(lefthandcounter), dict(righthandcounter))


def build_raw_text(corpus_list):
    raw = ""
    for word in corpus_list:
        raw += word + ' '
    return raw


words1freqs, words2freqs = prep_corpus(words1, words2)

corp1 = build_raw_text(words1)
corp2 = build_raw_text(words2)


fulltextValence, fulltextFvec = emotion(corp1, labMT, shift=True, happsList=labMTvector)
temp = stopper(fulltextFvec, labMTvector, labMTwordList, stopVal=1.0)
avg = emotionV(temp, labMTvector)
print(avg)


# hardcode for the moment to save time
# avg = 5.720768099602948




sentiment_shift = sh.WeightedAvgShift(type2freq_1=words1freqs,
                                      type2freq_2=words2freqs,
                                      reference_value = avg,
                                      type2score_1='labMT_English',
                                      stop_lens=[(4,6)])


sentiment_shift.get_shift_graph(system_names = ['Seasons 1-2', 'Seasons 3-4'])



#### Part C ####
fulltextValence, fulltextFvec = emotion(corp2, labMT, shift=True, happsList=labMTvector)
temp = stopper(fulltextFvec, labMTvector, labMTwordList, stopVal=1.0)
avg = emotionV(temp, labMTvector)
print(avg)



sentiment_shift = sh.WeightedAvgShift(type2freq_1=words1freqs,
                                      type2freq_2=words2freqs,
                                      reference_value = avg,
                                      type2score_1='labMT_English',
                                      stop_lens=[(4,6)])


sentiment_shift.get_shift_graph(system_names = ['Seasons 1-2', 'Seasons 3-4'])






#### Part E ####
sentiment_shift = sh.WeightedAvgShift(type2freq_1=words1freqs,
                                      type2freq_2=words2freqs,
                                      type2score_1='labMT_English',
                                      reference_value=5,
                                      stop_lens=[(4.5,5.5)])


sentiment_shift.get_shift_graph(system_names = ['Seasons 1-2', 'Seasons 3-4'])



