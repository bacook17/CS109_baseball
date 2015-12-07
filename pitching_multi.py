#Loading in appropriate packages
import numpy as np, matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy import stats
import multiprocessing as mp
from sklearn.cross_validation import *
# In[2]:

#Loading in Relevant Dataframes
master = pd.read_csv('data/lahman/Master.csv')
pitching = pd.read_csv('data/lahman/Pitching.csv')
smalldf = pd.read_csv('data/small.csv')


retro_to_lah = dict(zip(master['retroID'], master['playerID']))

def pitcher_sim(p1_matchups, p2_matchups, n_common=0):
    p1name=list(p1_matchups['pID'][:1])[0]  #getting the IDs for both pitchers
    p2name=list(p2_matchups['pID'][:1])[0]
    dfhands=pd.DataFrame({'RL':[list(p1_matchups['RL'][:1])[0],list(p2_matchups['RL'][:1])[0]]}) #retrieving handedness from  smalldf
    listed= [p1name,p2name]
    two_pitcherIDs= [retro_to_lah[name] for name in listed] #getting the lahman ids for the 2 pitcher ids to use in Pitching Dataframe

    grouped2 =  pitching.groupby('playerID').sum().reset_index() #getting the summed data for each unique playerID
    mask2 = np.in1d(grouped2.playerID, two_pitcherIDs) #creating a mask for the 2 pitcher IDs needed
    pitchers=grouped2[mask2].reset_index() #resetting the index to 0,1 to make concatenating dataframes easier
    pitchers.ERA = 9.*(pitchers.ER)/(pitchers.IPouts / 3.) #generating ERA 
    pitchers.BAOpp = (pitchers.H + pitchers.BB + pitchers.IBB + pitchers.HBP) / pitchers.BFP #batting average against
    pitchers['WiP'] = pitchers.W/pitchers.G #win percentage
    pitchers['IP'] = pitchers.IPouts/3. #innings pitched
    
    #determining whether a pitcher is a relief pitcher based on whether more games started than not 
    #and less than 4 innings pitched per game
    pitchers['Games']=pitchers['GS']-pitchers['G']/2. #if positive, more games started than games in relief
    pitchers['inningsper']=pitchers['IP']/pitchers['G'] #innings per game 
    relief=[]
    for i in pitchers.iterrows():
        if i[1]['Games']<0. and i[1]['inningsper']<4.:
            relief.append(1.)
        else:
            relief.append(0.)
    dfrel=pd.DataFrame({'REL':relief})
    
    #adding handedness and whether a pitcher is a relief pitcher to the dataframe 
    final=pitchers.join([dfhands,dfrel], how='outer')
    
    #getting the 2 rows from the final dataframe to perform calculations 
    p1=final[final.playerID==two_pitcherIDs[0]]
    p2=final[final.playerID==two_pitcherIDs[1]]
    
    
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
    if list(p1['RL'])[0]!=list(p2['RL'])[0] and int(p1['REL'])==0 and int(p2['REL'])==0: #handedness is different, relief pitchers
        hand_diff=10. 
    elif list(p1['RL'])[0]!=list(p2['RL'])[0] and int(p1['REL'])==1 and int(p2['REL'])==1.: #handedness is different, starters
        hand_diff=25.  
    else:
        hand_diff=0.
    
    #Additional Penalties
    gp_diff=np.abs(int(p1['G'])-int(p2['G']))/10. #1 pt for diff of 10 games played
    st_diff=np.abs(int(p1['GS'])-int(p2['GS']))/20. #1 pt for diff of 20 games started
    cg_diff=np.abs(int(p1['CG'])-int(p2['CG']))/20. #1 pt for diff of 20 games completed
    ip_diff=np.abs(float(p1['IP'])-float(p2['IP']))/50. #1 pt for diff of 50 innings pitched
    h_diff=np.abs(int(p1['H'])-int(p2['H']))/50. #1 pt for diff of 50 hits allowed
    sp_diff=np.abs(int(p1['SO'])-int(p2['SO']))/30. #1 pt for diff of 30 strikeouts
    bb_diff=np.abs(int(p1['BB'])-int(p2['BB']))/50. #1 pt for diff of 10 walks
    sho_diff=np.abs(int(p1['SHO'])-int(p2['SHO']))/5. #1 pt for diff of 5 shutouts
    sv_diff=np.abs(int(p1['SV'])-int(p2['SV']))/3. #1 pt for diff of 3 saves
    

    score=1000 #generating starting score and subtracting penalties
    final_score=score-(win_diff+loss_diff+wip_diff+era_diff+gp_diff+st_diff+cg_diff+
                       ip_diff+h_diff+sp_diff+bb_diff+sho_diff+sv_diff+hand_diff)
    #print final_score

    return final_score


def get_restaurant_reviews(pID, df, set_of_batters):
    """
    given a pitcher id and a set of batters, return the sub-dataframe of their
    averages.
    """
    mask = (df.bID.isin(set_of_batters)) & (df.pID==pID)
    avgs = df[mask]
    avgs = avgs[avgs.bID.duplicated()==False]
    return avgs

trainlist=[]
testlist=[]
validatelist=[]
take=21 #21 matchups between validation and test set
for k, v in smalldf.groupby('bID'):
    if len(v) > 100: #batter has faced at least 150 pitchers
        train_rows, test_valid_rows = train_test_split(v, test_size=take)
        trainlist.append(train_rows)
        valid_rows, test_rows = train_test_split(test_valid_rows, test_size=0.4)
        validatelist.append(valid_rows) 
        testlist.append(test_rows) 
    else:
        trainlist.append(v)
traindf=pd.concat(trainlist)
validatedf=pd.concat(validatelist)
testdf=pd.concat(testlist)
print traindf.shape, validatedf.shape, testdf.shape


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

traindf=recompute_frame(traindf)
validatedf=recompute_frame(validatedf)
testdf=recompute_frame(testdf)
validatedf=validatedf[['bID', 'pID','AVG']]
testdf=testdf[['bID', 'pID', 'AVG']]

def compute_supports(df, upids):
    ubids=df.bID.unique()
    pitch = df.groupby('pID').bID.unique()
    bdict={}
    for e,v in zip(pitch.index.values, pitch.values):
        bdict[e] = np.array([item in v for item in ubids])
    pitchers=upids
    supports=[]
    supports_matrix = [[[] for i in range(len(pitchers))] for j in range(len(pitchers))]
    for i,p1 in enumerate(pitchers):
        for j,p2 in enumerate(pitchers):
            if  i < j:
                common_batters = set(pitch[p1]).intersection(set(pitch[p2]))
                n = len(common_batters)
                supports.append(n)
                supports_matrix[i][j] = common_batters
                supports_matrix[j][i] = common_batters
    print "mean support",np.mean(supports), "median support", np.median(supports)
    return supports, bdict, supports_matrix

ubids=traindf.bID.unique()#unique-user-ids
upids=traindf.pID.unique()#unique-item-ids
ubidmap={v:k for k,v in enumerate(ubids)}#of length U
upidmap={v:k for k,v in enumerate(upids)}#of length M


s,d, supports_matrix =compute_supports(traindf, upids)

class Database:
    "A class representing a database of similarities and common supports"
    
    def __init__(self, rindexmap, supports):
        "the constructor, takes a map of restaurant id's to integers"
        database={}
        self.rindexmap=rindexmap
        self.supports=supports
        l_keys=len(self.rindexmap.keys())
        self.database_sim=np.zeros([l_keys,l_keys])
        self.database_sup=np.zeros([l_keys, l_keys], dtype=np.int)

    def set_supports(self, supports):
        self.supports=supports
        
    def get(self, b1, b2):
        "returns a tuple of similarity,common_support given two business ids"
        sim=self.database_sim[self.rindexmap[b1]][self.rindexmap[b2]]
        nsup=self.database_sup[self.rindexmap[b1]][self.rindexmap[b2]]
        return (sim, nsup)


db=Database(upidmap, supports_matrix)


# In[87]:

def get_restaurant_reviews(pID, df, set_of_batters):
    """
    given a pitcher id and a set of batters, return the sub-dataframe of their
    averages.
    """
    mask = (df.bID.isin(set_of_batters)) & (df.pID==pID)
    avgs = df[mask]
    avgs = avgs[avgs.bID.duplicated()==False]
    return avgs


# In[88]:

def calculate_similarity(db, df, p1, p2, similarity_func):
    # find common reviewers
    common_reviewers = db.supports[db.rindexmap[p1]][db.rindexmap[p2]]
    n_common=len(common_reviewers)
    if p1==p2:
        return 1., n_common
    #get reviews
    if n_common==0:
        return 0., n_common
    p1_rows = get_restaurant_reviews(p1, df, common_reviewers)
    p2_rows = get_restaurant_reviews(p2, df, common_reviewers)
    sim=similarity_func(p1_rows, p2_rows, n_common)
    return sim, n_common


# In[ ]:

def populate_by_calculating_old(db, df, similarity_func):
    """
    a populator for every pair of businesses in df. takes similarity_func like
    pearson_sim as argument
    """
    items=db.rindexmap.items()
    for b1, i1 in items:
        for b2, i2 in items:
            count += 1
            if (((10000*count)/total) > ((10000*(count-1))/total)):
                frac_done = count/float(total)
                time_elapsed = (time.time() - start) / (60.)
                remaining = ((1./frac_done)-1.)*time_elapsed
                print('%.3f percent done | elapsed: %.1f min | remaining: %.2f hour'%(frac_done *100, time_elapsed, remaining/60))
            if i1 <= i2:
                sim, nsup=calculate_similarity(db, df, b1, b2, similarity_func)
                db.database_sim[i1][i2]=sim
                db.database_sim[i2][i1]=sim
                db.database_sup[i1][i2]=nsup
                db.database_sup[i2][i1]=nsup

def get_results(db, df, b1, b2, i1, i2, similarity_func):
    if i1 <= i2:
        sim, nsup=calculate_similarity(db, df, b1, b2, similarity_func)
        return sim,nsup
    else:
        return 0., 0
    
def populate_by_calculating_new(db, df, similarity_func):
    """
    a populator for every pair of businesses in df. takes similarity_func like
    pearson_sim as argument
    """
    items=db.rindexmap.items()
    pool = mp.Pool(processes=64)
    results = [[pool.apply_async(get_results, args=(db, df, b1, b2, i1, i2, similarity_func)) for b1, i1 in items] for b2, i2 in items]
    for b1, i1 in items:
        for b2, i2 in items:
            sim, nsup = results[i1][i2].get()
            db.database_sim[i1][i2]=sim
            db.database_sim[i2][i1]=sim
            db.database_sup[i1][i2]=nsup
            db.database_sup[i2][i1]=nsup
               
populate_by_calculating_old(db, traindf, pitcher_sim)
pickle.dump(db,open('db_pitcher.p','wb'))
