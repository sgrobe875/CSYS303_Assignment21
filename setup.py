# Sarah Grobe
# POCS Assignment 21
# Due 3/10





# Question 1 A
# deconstructing text into file w/ vector of 1-grams


# start by reading in the raw data
with open('data/stranger_things.txt') as f:
    data = f.readlines()
    

# check all of the nonletters
not_letters = set()
for item in data:
    for char in item:
        if not char.isalpha():
            not_letters.add(char)
            
# convert to a list so we can access indices
not_letters = list(not_letters)

# lists to hold the one-grams
one_grams = []    # final list
step1 = []        # intermediary list 

# loop through each line in the raw data
for line in data:
    # ignore the newline
    line = line[:-1]
    # split at each space
    line_sp = line.split(' ')
    # add each to step1 list
    for item in line_sp:
        step1.append(item)
        
        
        
# at this point, step1 holds list of all one-grams with the punctuation attached
        
clean_only = []
    
# loop through again to separate the punctuation/non-letters
for one_gram in step1:
    # if the 1-gram contains a non-alphabetic character
    if not one_gram.isalpha():
        # find the non-alphabetic character in not_letters and add to the list separately
        
        # also clean up the words to remove the punctuation, numbers, etc. from them
        cleaned = ""
        
        # loop through each character in the item
        for char in one_gram:
            # if the character in question is not a letter
            if char in not_letters:
                # find its location in not_letters and add that item to the one-grams list
                index = not_letters.index(char)
                one_grams.append(not_letters[index])
                
            # if it is a letter, append to the cleaned version of the word
            else:
                cleaned += char
                
        # also add the cleaned up word itself
        one_grams.append(cleaned)
        clean_only.append(cleaned)
        
    # if no non-alpha characters, then we just add the item as is
    else:
        one_grams.append(one_gram)
        clean_only.append(one_gram)



# write results to file so we don't have to do this every time
f = open("data/one_grams.txt", "w")
for item in one_grams:
    f.write(item + '\n')
f.close()



# alternatively, a data file which ignores the punctuation
f = open("data/one_grams_clean.txt", "w")
for item in clean_only:
    f.write(item + '\n')
f.close()





# split into first half and second half
cutoff = int(len(clean_only)/2)
f = open("data/one_grams_clean_first.txt", "w")
for i in range(cutoff):
    item = clean_only[i]
    f.write(item + '\n')
f.close()
    
    
f = open("data/one_grams_clean_second.txt", "w")
for i in range(cutoff, len(clean_only)):
    item = clean_only[i]
    f.write(item + '\n')
f.close()
    
    







