import math
import re
from strsimpy.qgram import QGram
from textdistance import levenshtein
from collections import Counter
import Levenshtein

def qgrams(s, q=3):
    if s is None:
        return []
    s = s.lower()
    max_index = len(s) - q + 1
    if max_index <= 0:
        return set()
    return {s[i:i + q] for i in range(max_index)}

#calcs the q-gram similarity between two strings
def qgram_similarity(s1, s2, q=3):
    q1 = qgrams(s1, q) #set of qgrams first string
    q2 = qgrams(s2, q) #set of qgrams second string
    if not q1 or not q2:
        return 0.0  # if both strings are empty then no info so 0
    overlap = len(q1.intersection(q2))
    total = len(q1.union(q2))
    return overlap / total

def qgram_kvp_similarity(kvp_i, kvp_j, gamma=0.7, q=3):

    nmki = dict(kvp_i)  # copy input dictionaries so dont change og, to keep track of unmatched kv pairs
    nmkj = dict(kvp_j)

    sim = 0.0 #cumulative sum of weighted VALUE sims
    w = 0.0 #cumulative sum of weights (the key similarities)
    m = 0 #counter for number of key matches

    for key_i, val_i in kvp_i.items(): #dubbel loop over all combinations of k-v pairs of product i and j
        for key_j, val_j in kvp_j.items():
            keySim = qgram_similarity(key_i, key_j, q) #calcs similarity of the key of i and j
            if keySim > gamma: #if this is above threshold, then calc similarity of the values
                valueSim = qgram_similarity(val_i, val_j, q) #calc qgram sim of the values belonging to these keys
                weight = keySim
                sim += weight * valueSim #calc the weighted value similarity and add to cum sum
                m += 1 #add a key match
                w += weight #add the key similarity weight

                #needed as input for HSM: delete matched kv pairs (not products) from the unmatched sets, because matched now
                nmki.pop(key_i, None) #delete key from dictionary prod i
                nmkj.pop(key_j, None) #delete key from dictionary prod j

    #calc avgSim
    if w > 0:
        avgSim = sim / w
    else:
        avgSim = 0.0
    return avgSim, m, nmki, nmkj #return this as input for HSM

#HSM
from mw import extract_model_words_value_dict #function to extract model words

# nmki and nmkj are product dictionaries excluding the key-value pairs already processed in the q-gram step
def hsm_similarity(nmki, nmkj):
    mw_i = extract_model_words_value_dict(nmki)  #set of model words product i of values that are not matched
    mw_j = extract_model_words_value_dict(nmkj)  #same for j

    #if one set empty, no similarity
    if not mw_i or not mw_j:
        return 0.0
    intersection = mw_i.intersection(mw_j)
    union = mw_i.union(mw_j)
    percentage = len(intersection) / len(union) if len(union) !=0 else 0
    return percentage


#############################################
#TMWM
import math
from mw import extract_model_words_title

#calc cos sim betweem 2 product titles, EXTRA CLEANING NEEDED???
def cosine_title_similarity(title1, title2):
    words_title_1 = set(str(title1).split()) #split the words by spaces
    words_title_2 = set(str(title2).split())
    if not words_title_1 or not words_title_2:
        return 0.0
    intersection = len(words_title_1 & words_title_2)
    cossim = intersection / (math.sqrt(len(words_title_1)) * math.sqrt(len(words_title_2)))
    return cossim if len(words_title_1)!= 0 and len(words_title_2) != 0 else 0

#Split a model word into nonnumeric part and numeric part (used to check if non-num parts are similar and numeric parts are equal)
def split_model_word(mw):
    mw = str(mw)
    non_num_part = re.sub(r"\d+", "", mw)
    num_part = "".join(re.findall(r"\d+", mw))
    return non_num_part, num_part

def normalized_levenshtein(mw1, mw2):
    mw1 = "" if mw1 is None else str(mw1)
    mw2 = "" if mw2 is None else str(mw2)
    if not mw1 and not mw2:
        return 0.0
    maxl = max(len(mw1), len(mw2))
    if maxl == 0:
        return 0
    dist = Levenshtein.distance(mw1, mw2)
    norm_lv_dist = dist/maxl
    return norm_lv_dist

#Avg lv distance
#weighted average of (1 - normalized_levenshtein) over all word pairs.
def avg_Lv_Sim(words1, words2):

    words1 = list(words1)
    words2 = list(words2)
    if not words1 or not words2:
        return 0.0

    num = 0.0
    den = 0.0
    for x in words1:
        for y in words2:
            norm_lv = normalized_levenshtein(x, y)
            sim = 1.0 - norm_lv
            w = len(x) + len(y)
            num += sim * w
            den += w
    return num / den if den > 0 else 0.0

#like avglvsim but only over pairs where nonnum parts are approx same and numerical parts are exactly same
def avg_lv_sim_mw(mws1, mws2, approx_thresh=0.5):
    mws1 = list(mws1)
    mws2 = list(mws2)
    if not mws1 or not mws2:
        return 0.0

    num = 0.0
    den = 0.0
    for x in mws1:
        non_x, num_x = split_model_word(x)
        for y in mws2:
            non_y, num_y = split_model_word(y)

            #similarity between non-num parts
            approx_sim = 1.0 - normalized_levenshtein(non_x, non_y)
            if approx_sim <= approx_thresh:
                continue
            # numeric parts must be equal
            if num_x != num_y:
                continue

            norm_lv = normalized_levenshtein(x, y)
            sim = 1.0 - norm_lv
            w = len(x) + len(y)
            num += sim * w
            den += w

    return num / den if den > 0 else 0.0



#now calc the tmwm similarity
from mw import extract_model_words_title
def tmwm_similarity(title_i,title_j,alpha,beta,delta,approx_thresh=0.5):

    #cos sim of full titles
    name_cos_sim = cosine_title_similarity(title_i, title_j)

    #if high enough, treat as identical, sim =1
    if name_cos_sim > alpha:
        return 1.0

    #title mws using prev defined function
    mw_i = extract_model_words_title(str(title_i))
    mw_j = extract_model_words_title(str(title_j))

    #check if non-numeric parts similar, numeric parts different, then -1
    for x in mw_i:
        non_x, num_x = split_model_word(x)
        for y in mw_j:
            non_y, num_y = split_model_word(y)
            approx_sim = 1.0 - normalized_levenshtein(non_x, non_y)
            if approx_sim > approx_thresh and num_x != num_y:
                #do not use title in final similarity
                return -1.0

    #base imilarity using cosine and average Levenshtein over model words
    base_mw_sim = avg_Lv_Sim(mw_i, mw_j)
    final_sim = beta * name_cos_sim + (1.0 - beta) * base_mw_sim

    #update
    mw_sim_weighted = avg_lv_sim_mw(mw_i, mw_j, approx_thresh)

    if mw_sim_weighted > 0.0:
        #reweight with delta
        final_sim = delta * mw_sim_weighted + (1.0 - delta) * final_sim

    return final_sim

################################################
#MSM final weighted similarity measure
def msm_similarity(pi_idx,pj_idx,data,gamma=0.7,q=3, alpha=0.6,beta=0.0, delta=0.5,mu=0.65,approx_thresh=0.5):

    #get the kv dictionaries and titles for both products
    kvp_i = data.iloc[pi_idx]["featuresMap"]
    kvp_j = data.iloc[pj_idx]["featuresMap"]
    title_i = data.iloc[pi_idx]["title"]
    title_j = data.iloc[pj_idx]["title"]
    if not isinstance(kvp_i, dict):
        kvp_i = {}
    if not isinstance(kvp_j, dict):
        kvp_j = {}

    #1) Q-gram similarity on matching KVPs
    avgSim, m, nmki, nmkj = qgram_kvp_similarity(kvp_i, kvp_j, gamma=gamma, q=q)

    #2) HSM: model words from non matched kvps
    mwPerc = hsm_similarity(nmki, nmkj)

    #3)  TMWM
    titleSim = tmwm_similarity(title_i,
                               title_j,
                               alpha=alpha,
                               beta=beta,
                               delta=delta,
                               approx_thresh=approx_thresh)

    if kvp_i or kvp_j:
        min_features = min(len(kvp_i), len(kvp_j))
    else:
        min_features = 0

    if min_features > 0:
        key_match_ratio = m / float(min_features)
    else:
        key_match_ratio = 0.0

    if titleSim == -1:
        #titleSim is not used, so indicator is zero
        theta1 = key_match_ratio
        theta2 = 1.0 - theta1
        hSim = theta1 * avgSim + theta2 * mwPerc
    else:
        #title sim used with weight mu
        theta1 = (1.0 - mu) * key_match_ratio
        theta2 = 1.0 - mu - theta1
        hSim = theta1 * avgSim + theta2 * mwPerc + mu * titleSim

    hSim = max(0.0, min(1.0, hSim))

    return hSim