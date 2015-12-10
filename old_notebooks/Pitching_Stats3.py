#Loading in appropriate packages
import numpy as np, matplotlib.pyplot as plt
import pandas as pd
import requests
import time
import json
from bs4 import BeautifulSoup
import sys
import multiprocessing as mp
from scipy import stats
from sklearn.cross_validation import *

master = pd.read_csv('data/lahman/Master.csv')
pitching = pd.read_csv('data/lahman/Pitching.csv')
smalldf = pd.read_csv('data/small.csv')


#Ben's intro code
retro_to_lah = dict(zip(master['retroID'], master['playerID']))
lah_to_retro = dict(zip(master['playerID'], master['retroID']))
retro_to_hand = dict(zip(master['retroID'], master['throws']))
def recompute_frame(ldf):
    """
    takes a dataframe ldf, makes a copy of it, and returns the copy
    with all conglomerations recomputed
    this is used when a frame is subsetted.
    """
    ldfb=ldf.groupby('bID')
    ldfp=ldf.groupby('pID')
    nldf=ldf.copy()
    
    #Conglomerate pitcher stats
    nldf.set_index(['pID'], inplace=True)
    for col in ['AB', 'PA', 'H', 'TB', 'SAC', 'SO', 'W']:
        nldf['ovp_'+col] = ldfp[col].sum()
    nldf['ovp_AVG'] = nldf['ovp_H']/nldf['ovp_AB']
    nldf['ovp_FACED']= ldfp.AB.count()
    nldf['ovp_OBP'] = (nldf['ovp_H'] + nldf['ovp_W'])/nldf['ovp_PA']
    for col in ['SO', 'W', 'H']:
        nldf['ovp_' + col + '_PCT'] = nldf['ovp_' + col] / nldf['ovp_PA']
    nldf.reset_index(inplace=True)
    
    #Conglomerate batter stats
    nldf.set_index(['bID'], inplace=True)
    for col in ['AB', 'PA', 'H', 'TB', 'SAC', 'SO', 'W']:
        nldf['ov_'+col] = ldfb[col].sum()
    nldf['ov_AVG'] = nldf['ov_H']/nldf['ov_AB']
    nldf['ov_FACED']= ldfb.AB.count()
    nldf['ov_OBP'] = (nldf['ov_H'] + nldf['ov_W'])/nldf['ov_PA']
    for col in ['SO', 'W', 'H']:
        nldf['ov_' + col + '_PCT'] = nldf['ov_' + col] / nldf['ov_PA']
    nldf.reset_index(inplace=True)
    return nldf


##input is the smalldf rows corresponding to the 2 pitcher ids
def pitcher_sim_alt(p1, p2):
    #Wins and Losses Penalties
    win_diff=np.abs(int(p1['W'])-int(p2['W'])) #1 pt for each win
    loss_diff=np.abs(int(p1['L'])-int(p2['L']))/2. #1 pt each 2 losses
    
    #Winning Percentage Penalty
    if int(p1['REL'])==1:     #winning percentage is halved for relief pitchers
        p1['WiP']=p1['WiP']/2.
    if int(p2['REL'])==1:
        p2['WiP']=p2['WiP']/2.
    wip_diff=np.abs(float(p1['WiP'])-float(p2['WiP']))/.002 #1 pt diff of winning percentage of 0.002 
    if wip_diff > 1.5*(win_diff+loss_diff): #winning percentage cannot be more than 1.5 times the penalties for win/loss differences 
        wip_diff= 1.5*(win_diff+loss_diff)
    if wip_diff >100: #penalty for winning percentage has a max= 100
        wip_diff=100   
    era_diff=np.abs(float(p1['ERA'])-float(p2['ERA']))/0.02 #1 pt diff of ERA of 0.02
    if era_diff >100:
        era_diff=100 #max =100
    
    #Handedness Penalties conditional on both pitchers being relief pitchers or not 
    if p1['RL'].values[0]!=p2['RL'].values[0] and int(p1['REL'])==0 and int(p2['REL'])==0: #handedness is different, relief pitchers
        hand_diff=10. 
    elif p1['RL'].values[0]!=p2['RL'].values[0] and int(p1['REL'])==1 and int(p2['REL'])==1.: #handedness is different, starters
        hand_diff=25.  
    else:
        hand_diff=0.
    
    #Additional Penalties
    gp_diff=np.abs(int(p1['G'])-int(p2['G']))/10. #1 pt for diff of 10 games played
    st_diff=np.abs(int(p1['GS'])-int(p2['GS']))/20. #1 pt for diff of 20 games started
    cg_diff=np.abs(int(p1['CG'])-int(p2['CG']))/20. #1 pt for diff of 20 games completed
    ip_diff=np.abs(float(p1['IP'])-float(p2['IP']))/50. #1 pt for diff of 50 innings pitched
    h_diff=np.abs(int(p1['H'])-int(p2['H']))/50. #1 pt for diff of 50 hits allowed
    so_diff=np.abs(int(p1['SO'])-int(p2['SO']))/30. #1 pt for diff of 30 strikeouts
    bb_diff=np.abs(int(p1['BB'])-int(p2['BB']))/50. #1 pt for diff of 10 walks
    sho_diff=np.abs(int(p1['SHO'])-int(p2['SHO']))/5. #1 pt for diff of 5 shutouts
    sv_diff=np.abs(int(p1['SV'])-int(p2['SV']))/3. #1 pt for diff of 3 saves

    score=1000 #generating starting score and subtracting penalties
    final_score=score-(win_diff+loss_diff+wip_diff+era_diff+gp_diff+st_diff+cg_diff+
                       ip_diff+h_diff+so_diff+bb_diff+sho_diff+sv_diff+hand_diff)
    #print final_score

    return final_score



trainlist=np.array([])
testlist=np.array([])
validatelist=np.array([])
take=21 #21 matchups between validation and test set
for k, v in smalldf.groupby('bID'):
    if len(v) > 100: #batter has faced at least 150 pitchers
        train_rows, test_valid_rows = train_test_split(v.matchID.values, test_size=take)
        trainlist = np.append(trainlist,train_rows)
        valid_rows, test_rows = train_test_split(test_valid_rows, test_size=0.4)
        validatelist =np.append(validatelist,valid_rows) 
        testlist = np.append(testlist,test_rows) 
    else:
        trainlist = np.append(trainlist,v.matchID.values)
mask = np.in1d(smalldf.matchID.values, trainlist)
traindf=smalldf[mask]
mask = np.in1d(smalldf.matchID.values, validatelist)
validatedf=smalldf[mask]
mask = np.in1d(smalldf.matchID.values, testlist)
testdf=smalldf[mask]

traindf=recompute_frame(traindf)
validatedf=recompute_frame(validatedf)
testdf=recompute_frame(testdf)
validatedf=validatedf[['bID', 'pID','AVG']]
testdf=testdf[['bID', 'pID', 'AVG']]
traindf.head()


ubids=traindf.bID.unique()#unique-user-ids
upids=traindf.pID.unique()#unique-item-ids
ubidmap={v:k for k,v in enumerate(ubids)}#of length U
upidmap={v:k for k,v in enumerate(upids)}#of length M

def get_pitching_totals(pitching_df, upids):
    lah_pitchers = [retro_to_lah[name] for name in upids]
    pitching_totals = pitching.groupby('playerID').sum().reset_index()
    pitching_totals = pitching_totals[np.in1d(pitching_totals['playerID'], lah_pitchers)]
    pitching_totals.head()
    pitching_totals['BB_total'] = pitching_totals['BB'] + pitching_totals['IBB'] + pitching_totals['HBP']
    pitching_totals['pID'] = [lah_to_retro[name] for name in pitching_totals.playerID]
    pitching_totals['RL'] = [retro_to_hand[name] for name in pitching_totals.pID]
    pitching_totals['IP'] = pitching_totals['IPouts']/3.
    pitching_totals['Games']=pitching_totals['GS']-pitching_totals['G']/2. #if positive, more games started than games in relief
    pitching_totals['inningsper']=pitching_totals['IP']/pitching_totals['G'] #innings per game 
    relief=[1 if (i[1]['Games'] < 0. and i[1]['inningsper']<4.) else 0 for i in pitching_totals.iterrows()]
    pitching_totals['REL'] = relief
    pitching_totals['ERA'] = pitching_totals['ER'] *9. / (pitching_totals['IP'])
    pitching_totals['BAOpp'] = (pitching_totals.H + pitching_totals.BB_total) / pitching_totals.BFP #batting average against
    pitching_totals['WiP'] = pitching_totals.W/pitching_totals.G #win percentage

    return pitching_totals[['pID', 'W', 'L', 'ERA', 'RL', 'BAOpp', 'WiP', 'IP', 'REL', 'G', 'GS', 'CG', 'H', 'SO', 'BB', 'SHO', 'SV']]


pitching_totals = get_pitching_totals(pitching, upids)



class Database_pitch:
    "A class representing a database of similarities and common supports"
    
    def __init__(self, rindexmap):
        "the constructor, takes a map of restaurant id's to integers"
        database={}
        self.rindexmap=rindexmap
        self.inversemap = dict(zip(self.rindexmap.values(), self.rindexmap.keys()))
        l_keys=len(self.rindexmap.keys())
        self.database_sim=np.zeros([l_keys,l_keys])
        
    def get(self, b1, b2):
        "returns a tuple of similarity,common_support given two business ids"
        sim=self.database_sim[self.rindexmap[b1]][self.rindexmap[b2]]
        return sim


db=Database_pitch(upidmap)

def populate_by_calculating(db, df, similarity_func):
    """
    a populator for every pair of businesses in df. takes similarity_func like
    pearson_sim as argument
    """
    items=db.rindexmap.items()
    start = time.time()
    count = 0
    total = len(items)**2

    for p1, i1 in items:
        for p2, i2 in items:
            count += 1
            if (((100*count)/total) > ((100*(count-1))/total)):
                frac_done = (count+1)/float(total)
                time_elapsed = (time.time() - start) / (60.)
                remaining = ((1./frac_done)-1.)*time_elapsed
                print('%.1f percent done | elapsed: %.1f min | remaining: %.2f hour'%(frac_done *100, time_elapsed, remaining/60))
            if i1 <= i2:
                p1_row = df[df.pID==p1]
                p2_row = df[df.pID==p2]
                sim =similarity_func(p1_row, p2_row)
                db.database_sim[i1][i2]=sim
                db.database_sim[i2][i1]=sim

def populate_by_calculating_multi(db, df, similarity_func, n=1):
    """
    a populator for every pair of businesses in df. takes similarity_func like
    pearson_sim as argument
    """
    items=db.inversemap.items()[:200]
    start = time.time()
    count = 0
    total = len(items)**2

    pool=mp.Pool(processes=n)
    results = [[pool.apply_async(similarity_func, args=(df[df.pID==p1],df[df.pID==p2])) if i1<=i2 else 0. for i2, p2 in items] for i1, p1 in items]
    for i1, p1 in items:
        for i2, p2 in items:
            if i1 <= i2:
                sim =results[i1][i2].get()
                db.database_sim[i1][i2]=sim
                db.database_sim[i2][i1]=sim

print('Starting')
populate_by_calculating(db, pitching_totals, pitcher_sim_alt)
# In[ ]:

import cPickle as pickle
pickle.dump(db,open('pitching_db.p','wb'))
