import numpy as np
import pandas as pd
import random
import altair as alt
import inspect
from IPython.display import display, Markdown
from toolz.itertoolz import groupby, sliding_window, partition
from toolz.dicttoolz import itemmap, valmap
from joblib import Parallel, delayed
from toolz.dicttoolz import valfilter

def singleChoice(poolName):
    def fn(aggRecord=None, tau=0):
        return poolName
    return {"fn":fn,
            "label": f"""SingleChoice{poolName}"""}

def randomChoice():
    def fn(aggRecord=None, tau=0):
        return random.choice(["S", "H", "L"])
    return {"fn":fn,
            "label": """RandomChoice"""}

#Whoops.  I should have made the Best Mean strats only subtract Tau from the other Pools
def bestMeanNBack(nBack):
    def fn(aggRecord=None, tau=0):
        means = aggRecord[['LPayoff', 'HPayoff', 'SPayoff']][-nBack:].mean()
        meanMinusTau = means - tau
        return meanMinusTau.sort_values().index[-1][0] #Grabs biggest value's label, returns first letter
    return {"fn":fn, "label": f"""BestMeanLookback{nBack}"""}


def bestMedianNBack(nBack):
    def fn(aggRecord=None, tau=0):
        means = aggRecord[['LPayoff', 'HPayoff', 'SPayoff']][-nBack:].median()
        meanMinusTau = means - tau
        return meanMinusTau.sort_values().index[-1][0] #Grabs biggest value's label, returns first letter
    return {"fn":fn, "label": f"""BestMedianLookback{nBack}"""}

def collective():
    def fn(aggRecord=None, tau=0):
        return "N/A"
    return {"fn":fn,
            "label": "Collective"}

def assignCollectiveStrats(board):
    collectiveIndices = [x[0] for x in board.items() 
                                    if x[1].updateStratLabel =="Collective"]
    
    
    
    board[collectiveIndices[0]].updateStratFn = singleChoice("H")["fn"]
    board[collectiveIndices[0]].updateStratLabel = "CollectiveH"
    
    board[collectiveIndices[1]].updateStratFn = singleChoice("L")["fn"]
    board[collectiveIndices[1]].updateStratLabel = "CollectiveL"
    
    for x in collectiveIndices[2:]:
        board[x].updateStratFn = singleChoice("S")["fn"]
        board[x].updateStratLabel = "CollectiveS"


def collectivePayout(board):
    collectiveIndices = [x[0] for x in board.items() 
                        if "Collective" in x[1].updateStratLabel]
    
    collectiveShare = sum(board[x].balance for x in collectiveIndices)/len(collectiveIndices)
    
    for x in collectiveIndices:
        board[x].balance = collectiveShare

#HERE
class agent(object):

    def __init__(self, name, startingStrat, updateStrat, switchThreshold=np.inf):
        """Return an agent whose name is *name* and starting
        balance is *balance*."""
        self.name = name
        self.balance = 0
        self.startingStratFn = startingStrat()["fn"]
        self.startingStratLabel = startingStrat()["label"]
        self.updateStratFn = updateStrat["fn"]
        self.updateStratLabel = updateStrat["label"]
        self.location = self.startingStratFn()
        if self.updateStratLabel == "Collective":
            self.switchThreshold = np.inf
        if self.updateStratLabel != "Collective":
            self.switchThreshold = switchThreshold
        self.origSwitchThreshold = switchThreshold
        
    def pay(self, amt):
        self.balance += amt
        
    def switchToMostProfitableStrat(self, scoresByStrats, board):
        bestStratLabel = (scoresByStrats
                            .groupby("Strategy")["Balance"]
                            .mean()
                            .sort_values(ascending=False)).index[0]
        bestStratFn = [x for x 
                       in board.values() 
                       if x.updateStratLabel == bestStratLabel][0].updateStratFn
        
        self.updateStratLabel = bestStratLabel
        self.updateStratFn = bestStratFn
        
    def updateLoc(self, aggRecord, tau, scoresByStrats, board):
        roundsPassed = aggRecord.shape[0]
        if roundsPassed > self.switchThreshold:
            self.switchToMostProfitableStrat(scoresByStrats, board)
            self.switchThreshold += self.origSwitchThreshold
        self.location = self.updateStratFn(aggRecord, tau)
        
    def __str__(self):
        return (f"Balance: {self.balance} "
               f"Location: {self.location} "
               f"Strategy: {self.updateStratLabel} "
               f"SwitchThrsh: {str(self.origSwitchThreshold)}")
        
    
    def __repr__(self):
        return (f"Balance: {self.balance} "
               f"Location: {self.location} "
               f"Strategy: {self.updateStratLabel} "
               f"SwitchThrsh: {str(self.origSwitchThreshold)}")

def assignRoundStats(board):
    LPop = sum(1 for x in board.values() if x.location=="L")
    HPop = sum(1 for x in board.values() if x.location=="H")
    SPop = sum(1 for x in board.values() if x.location=="S")
    
    if LPop == 0:
        LPayoff = np.random.choice((0,40), p=[0.5,0.5])
    if LPop>0:
        LPayoff = np.random.choice((0,40), p=[0.5,0.5]) / LPop

                                
    if HPop == 0:
        HPayoff = np.random.choice((0,80), p=[0.75,0.25])
    if HPop>0:
        HPayoff = np.random.choice((0,40), p=[0.75,0.25]) / HPop
        
    
        
    return dict(zip(["LPop", "LPayoff", "HPop", "HPayoff", "SPop", "SPayoff"], 
                    [LPop, LPayoff, HPop, HPayoff, SPop, 1]))

def payAgent(payoutDct, agent):
    if agent.location == "H":
        agent.pay(payoutDct["HPayoff"])
        
    if agent.location == "L":
        agent.pay(payoutDct["LPayoff"])
        
    if agent.location == "S":
        agent.pay(payoutDct["SPayoff"])

#Here
def oneStep(board, moveRecord, aggRecord, balanceRecord, stratRecord, tau=0):
    roundStats = assignRoundStats(board)
    
    for x in board.values():
        payAgent(roundStats, x)
        
    collectivePop = sum(1 for x in board.values() 
                 if "Collective" in x.updateStratLabel)
    
    if collectivePop>0:
        collectivePayout(board)
    
    aggRecord.loc[len(aggRecord)] = list(roundStats.values())
    
    #Get scores by strats so that agents that look at other agents
    #can see what's effective.  Exclude Collectives, cuz they work differently
    #Maybe make an alt version where people can join the Collective
    nonCollectives = valfilter(lambda x:  "Collective" not in x.updateStratLabel, board)
    scoresByStratsForUpdates = finalScore(nonCollectives)
    
    for x in board.values():
        x.updateLoc(aggRecord, tau, scoresByStratsForUpdates, board)
    
    movesThisRound = pd.DataFrame(x.location for x in board.values())
    
    moveRecord[str(moveRecord.columns.shape[0])] = movesThisRound[0]
    
    balancesThisRound = pd.DataFrame(x.balance for x in board.values())
    
    balanceRecord[str(balanceRecord.columns.shape[0])] = balancesThisRound[0]
    
    stratsThisRound = pd.DataFrame(x.updateStratLabel for x in board.values())
    
    stratRecord[str(stratRecord.columns.shape[0])] = stratsThisRound[0]

def agentToDict(agent):
    return dict(partition(2, repr(agent).replace(":","").split(" ")))

def finalScore(board):
    return (pd.DataFrame(valmap(agentToDict, board))
            .transpose()
            .assign(Balance = lambda x: x["Balance"].astype(float)))

#finalScore(sim["board"])

def runSimulation(startingStrat, updateStrats, switchThresholds=[np.inf], nAgents=50, nRound=100, stratProportions=None, tau=0):
    if not stratProportions:
        board = {x: agent(str(x), 
                      randomChoice, 
                      random.choice(updateStrats),
                         random.choice(switchThresholds)) 
             for x in range(nAgents)}
    if stratProportions:
        board = {x: agent(str(x), 
                      randomChoice, 
                      random.choice(updateStrats, p=stratProportions),
                         random.choice(switchThresholds)) 
             for x in range(nAgents)}
        
    collectivePop = sum(1 for x in board.values() 
                 if "Collective" in x.updateStratLabel)
    
    if collectivePop>0:
        assignCollectiveStrats(board)
        
        
    aggRecord = pd.DataFrame(columns=["LPop", "LPayoff", "HPop", "HPayoff", "SPop", "SPayoff"])
    
    moveRecord = pd.DataFrame(x.location for x in board.values())
    
    balanceRecord = pd.DataFrame(x.balance for x in board.values())
    
    stratRecord = pd.DataFrame(x.updateStratLabel for x in board.values())
    
    for x in range(nRound):
        oneStep(board, moveRecord, aggRecord, balanceRecord, stratRecord, tau)
        
    #Remember, agent Strats & Starting position are in the Final Score
        
    return {"board": board,
           "aggRecord": aggRecord,
           "moveRecord": moveRecord,
           "finalScores": finalScore(board),
           "balanceRecord":balanceRecord,
           "stratRecord":stratRecord}

def graphBoardPopulations(aggRecord):
    dfToUse = aggRecord.copy()
    dfToUse["round"] = dfToUse.index.astype(int)
    meltedRecordPops = dfToUse[['LPop', 'HPop', 'SPop', 'round']].melt(id_vars=["round"])
    display(alt.Chart(meltedRecordPops).mark_bar().encode(
        x='round:O',
        y='value:Q',
        color='variable:N',))
    
#graphBoardPopulations(sim["aggRecord"])

def graphBoardPopulationsJustGraph(aggRecord):
    dfToUse = aggRecord.copy()
    dfToUse["round"] = dfToUse.index.astype(int)
    meltedRecordPops = dfToUse[['LPop', 'HPop', 'SPop', 'round']].melt(id_vars=["round"])
    return alt.Chart(meltedRecordPops).mark_bar().encode(
        x='round:O',
        y=alt.Y('value:Q', title="Pop"),
        color=alt.Color('variable:N', legend=alt.Legend(title="Pool", orient="left"))).properties(
    width=700, 
    height = 180)

def graphPayoutsJustGraph(sim):
    #Make a version of this with pops
    aggRecord = sim["aggRecord"][['LPayoff', 'HPayoff']].copy()
    meltedAgg = aggRecord.assign(round= aggRecord.index).melt(id_vars="round")
    return alt.LayerChart(meltedAgg).encode(
            x='round',
            y='value:Q',
            color=alt.Color('variable:N', legend=alt.Legend(title="Payout", orient="left"))
        ).add_layers(
            alt.Chart().mark_point(),
            alt.Chart().mark_line()
        ).properties(
    width=700, 
    height = 180)

def graphPopsAndPayouts(sim):
    #display(Markdown("# Populations & Payouts"))
    display(graphBoardPopulationsJustGraph(sim["aggRecord"]))
    display(graphPayoutsJustGraph(sim))

def graphPopsByStrategy(sim, strat):
    simRecord = sim["moveRecord"].copy()
    simRecord["Strategy"] = sim["finalScores"]["Strategy"]
    stratLocCounts = (simRecord[simRecord["Strategy"]==strat]
     .iloc[:,:-1]
     .apply(lambda x: x.value_counts())
     .replace(np.nan, 0)
     .transpose()
     .rename(columns={'H': 'HPop', 'L': 'LPop', 'S': 'SPop'}))
    display(Markdown(f"### {strat}"))
    graphBoardPopulations(stratLocCounts)
    

def getScoresForStrats(sim):
    return (sim["finalScores"]
            .groupby("Strategy")["Balance"]
            .mean()
            .sort_values(ascending=False))

def getDescsForStrats(simDct):
    return (simDct["finalScores"]
            .groupby("Strategy")["Balance"]
            .describe())

def getDescsForSwitch(simDct):
    return (simDct["finalScores"]
            .groupby("SwitchThrsh")["Balance"]
            .describe())

def firstRoundWithoutStrat(sim, strat):
    stratCounts = stratCountsByRound(sim)
    if stratCounts[stratCounts[strat]==0].shape[0]==0:
        return stratCounts.shape[0]
    else:
        return int(stratCounts[stratCounts[strat]==0].index[0])
    

def firstRoundWithoutStrat(sim, strat):
    stratCounts = stratCountsByRound(x)
    if stratCounts[stratCounts[strat]==0].shape[0]==0:
        return stratCounts.shape[0]
    return int(stratCounts[stratCounts[strat]==0].index[0])
    

def stratScoresForMultSims(simList):
    return pd.DataFrame([getScoresForStrats(x) for x in simList]).describe().transpose()

def getWideStratScoresMultiSims(simList):
    return pd.DataFrame(getScoresForStrats(x) for x in simList)

def getPlaybackCountTable(sim):
    dfToUse = sim["moveRecord"].copy()
    dfToUse["round"] = sim["moveRecord"].index.astype(int)
    dfToUse["Strategy"] = sim["finalScores"]["Strategy"]
    meltedStrats = dfToUse.melt(id_vars=["round", "Strategy"],
                              value_name="Location")
    
    return pd.crosstab(meltedStrats["round"], [meltedStrats["Strategy"], meltedStrats["Location"]])

def multipleSims(nSims, startingStrat, updateStrats, switchThresholds=[np.inf], nAgents=50, nRound=100, stratProportions=None, tau=0):
    return [runSimulation(startingStrat, updateStrats, switchThresholds, nAgents, nRound, stratProportions, tau)
           for x in range(nSims)]

#tenSims = multipleSims(10, 
#                    randomChoice(), 
#                    [singleChoice("S")] + 
#                    [randomChoice()] + 
#                    [bestMeanNBack(x) for x in range(5)] +
#                    [bestMedianNBack(x) for x in range(5)],
#                    [np.inf] + list(range(5)))

def getScoresForStrats(simDct):
    return simDct["finalScores"].groupby("Strategy")["Balance"].mean()


def stratScoresForMultSims(simList):
    return (pd.DataFrame([getScoresForStrats(x) for x in simList])
            .describe()
            .transpose()
           .sort_values("mean", ascending=False))

def multipleSimsParallel(nSims, startingStrat, updateStrats, stratProportions=None, tau=0):
    return Parallel(n_jobs=-1)(delayed(runSimulation)(startingStrat, updateStrats, stratProportions, tau)
           for x in range(nSims))

#multipleSimsParallel(10, 
#                     randomChoice, 
#                    [randomChoice(), 
#                     bestMedianNBack(1), 
#                     bestMeanNBack(1)])

def graphAllStratPopsForSim(sim):
    for x in sim["finalScores"]["Strategy"].unique():
        graphPopsByStrategy(sim, x)

def getPoolPayoff(sim, poolName):
    aggRecord = sim["aggRecord"]
    return aggRecord[aggRecord[f"{poolName}Payoff"]>0]

#getPoolPayoff(sim, "H")

def getWideStratScoresMultiSims(simList):
    return pd.DataFrame(getScoresForStrats(x) for x in simList)

def multiSimStats(simList):
    wideScores = getWideStratScoresMultiSims(simList)

    wideScores["FirstHPayoff"] = [getPoolPayoff(x, "H").index[0] for x in simList]

    wideScores["FirstLPayoff"] = [getPoolPayoff(x, "L").index[0] for x in simList]

    wideScores.index = list(range(10))
    
    return wideScores

#Not that useful
def graphMultiSimStats(simList):
    wideScores = multiSimStats(simList)
    display(alt.Chart(wideScores).mark_circle().encode(
        alt.X(alt.repeat("column"), type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
    ).properties(
        width=250,
        height=250
    ).repeat(
        row=list(wideScores.columns[:-3].values),
        column=['FirstHPayoff', 'FirstLPayoff']
    ))
    
#graphMultiSimStats(tenSims)

def plotStrategyWealthOverTime(sim):
    simRecord = sim["balanceRecord"].copy()
    simRecord["Strategy"] = sim["finalScores"]["Strategy"]
    wideScores = (simRecord
     .groupby("Strategy")[simRecord.columns[:-1]]
     .mean()
     .transpose()
     .assign(round= range(101))
     .melt(id_vars="round"))
    
    display(alt.LayerChart(wideScores).encode(
        x='round',
        y='value:Q',
        color=alt.Color('Strategy:N', legend=alt.Legend(title="Payout", orient="left"))
    ).add_layers(
        alt.Chart().mark_point(),
        alt.Chart().mark_line()
    ).properties(
    width=700, 
    height = 180))
    
def plotMultipleWealthOverTime(simList):
    for x in simList:
        plotStrategyWealthOverTime(x)

def graphPayouts(sim):
    #Make a version of this with pops
    aggRecord = sim["aggRecord"][['LPayoff', 'HPayoff']].copy()
    meltedAgg = aggRecord.assign(round= aggRecord.index).melt(id_vars="round")
    display(alt.LayerChart(meltedAgg).encode(
            x='round',
            y='value:Q',
            color='variable:N'
        ).add_layers(
            alt.Chart().mark_point(),
            alt.Chart().mark_line()
        ))

def graphPayoutsAndAgentWealth(sim):
    plotStrategyWealthOverTime(sim)
    graphPayouts(sim)

def graphMultiPayoutsAgentWealth(simList):
    for x in simList:
        display(Markdown("# Game"))
        graphPayoutsAndAgentWealth(x)
        


def graphPayoutsAndAgentWealth(sim):
    plotStrategyWealthOverTime(sim)
    graphPayouts(sim)

def graphMultiPayoutsAgentWealth(simList):
    for x in simList:
        display(Markdown("# Game"))
        graphPayoutsAndAgentWealth(x)
        
def stratCountsByRound(sim):
    return (sim["stratRecord"]
              .apply(lambda x: x.value_counts())
              .replace(np.nan, 0)
              .transpose())

def graphStratCountsByRound(sim):
    stratCounts = stratCountsByRound(sim)
    dfToUse = stratCounts.copy()
    dfToUse["round"] = dfToUse.index.astype(int)
    meltedPops = dfToUse.melt(id_vars=["round"])
    display(alt.Chart(meltedPops).mark_bar().encode(
        x='round:O',
        y='value:Q',
        color=alt.Color('variable:N', legend=alt.Legend(title="Payout", orient="left"))).properties(
    width=700, 
    height = 180))
    
#graphStratCountsByRound(sim["stratRecord"])

#for x in tenSims:
#    graphStratCountsByRound(x)
#Maybe turn into function
#Maybe merge with graph of payouts