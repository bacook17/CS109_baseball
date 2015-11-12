# CS109_baseball
Final Project for CS 109 studying baseball statistics

#Contents

##Analysis
###DataScraping.ipynb
Generates a list (using the Lahman database) of all batters who appeared in 2004 or later.

Uses this list to search Retrosheet.org batter-pitcher matchup pages to compile a dataset with statistics for
each pitcher those batters faced at least 4 times.

Has some EDA figures to look at the distribution of At Bats, the distribution of pitchers faced, etc.

##Data
###lahman/*
Datafiles gathered from Lahman database. Most important are:

Batting (batting statistics by year)

Master (master details on each player, including Retrosheet ID).

###players.dat
FirstName,LastName,ID,AppearanceDate for all the players in the Retrosheet database

###batter_pages.json
Datadump of text from all Retrosheet pages for the list of batters. Many batters were not found.

###found.json
List of retroIDs of those batters for whom a Retrosheet matchup page was found.

###not_found.json
Those RetroIDs not found.

###matchups_2004.json
Database of batter-pitcher matchup pairs, including:

bID: batter ID

pID: pitcher ID

R/L: pitcher R/L handed

AB: number of at bats in matchup

PA: plate appearances

H: hits

TB: total bases

W: walks

SO: strikeouts

SAC: sacrifices
