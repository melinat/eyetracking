import numpy as np
import sys
import os.path
from eyeutil import *
import scipy.stats as sci
data_dir = "/home1/melina.tsitsiklis/eyetracking/behavioral/" #"/Users/anshpatel/Desktop/Freiburg_analysis/behavioral/"

subjects_id = ["Melina"] #["200218","150318","100418","240418","300518","050918","120918"]

# phase_dict={'INSTRUCTION_VIDEO': 1, 'HOMEBASE_TRANSPORT': 2, 'TRIAL_NAVIGATION':3,'PLAYER_CHEST_ROTATION':4,
#             'TOWER_TRANSPORT':5,'DISTRACTOR_GAME':6,'RECALL_PHASE':7,'LOCATION_RECALL_CHOICE':8,
#             'TEMPORAL_RETRIEVAL':9,'FEEDBACK':10,'SCORESCREEN':11,'BLOCKSCREEN':12,'TEMPORAL_FEEDBACK':13,'OBJECT_RECALL_CHOICE':14}

# phase_dict={'INSTRUCTION_VIDEO': 1,'HOMEBASE_TRANSPORT': 2,'TRIAL_NAVIGATION': 3, 'PLAYER_CHEST_ROTATION': 4,
#             'TOWER_TRANSPORT': 5, 'DISTRACTOR_GAME': 6, 'RECALL_PHASE': 7, 'RECALL_CHOICE': 8,
#             'FEEDBACK': 9, 'SCORESCREEN': 10}

ignore_phase={'RECALL_SPECIAL','SHOWING_INSTRUCTIONS','SPHINX_EVENT'}
def calculate_spearman_correlation(x,y):
	return sci.spearmanr(x,y)

def calculate_cortana_perf(cortanaPerf, totalCortana):
	cortanaAvg = np.zeros([40,1],dtype=float)
	for i in range(0,40):
		if totalCortana[i,0] > 0:
			cortanaAvg[i,0]=float(cortanaPerf[i,0])/float(totalCortana[i,0])
	return cortanaAvg

def compare_items(left,right,itemlist,trialindex,correct_response,subj_index,correct_index,correct,total_index,total):
	if right not in "":
		index=0
		leftindex=-1
		rightindex=-1
		# print(type(itemlist))
		for i in range(0,len(itemlist)):
			items=str(itemlist[i])
			# print(str(left) + " vs " + str(items))
			if left in items:
				leftindex=index
			if right in items:
				rightindex=index
			index+=1
		if leftindex<rightindex:
			if correct_response:
				correct_index[rightindex,subj_index]+=1
				correct[trialindex,subj_index]+=1
				# print("adding to total index at " + str(rightindex))
				total_index[rightindex,subj_index]+=1
			else:
				# print("adding to total index at " + str(rightindex))
				total_index[rightindex,subj_index]+=1
			total[trialindex,subj_index]+=1
		else:
			if correct_response:
				correct_index[leftindex,subj_index]+=1
				correct[trialindex,subj_index]+=1
				# print("adding to total index at " + str(leftindex))
				total_index[leftindex,subj_index]+=1
			else:
				# print("adding to total index at " + str(leftindex))
				total_index[leftindex,subj_index]+=1
			total[trialindex,subj_index]+=1
	else:
#         print left + " vs " + right
		return 0

def return_phase_index(phase_name, phase_dict):
	phase_name=phase_name.replace('_ENDED','')
	phase_name=phase_name.replace('_STARTED','')
	return phase_dict[phase_name]

def calculate_location_perf(data_dir,subjects_id):
	subj_count = len(subjects_id)
	print("subjects count is: " + str(subj_count))
	total=np.zeros([40,subj_count],dtype=int)
	perc=np.zeros([40,subj_count],dtype=float)
	correct_index=np.zeros([5,subj_count],dtype=int)
	total_index=np.zeros([5,subj_count],dtype=int)    
	correct=np.zeros([40,subj_count],dtype=int)
	perc_index=np.zeros([5,subj_count],dtype=float)
	last_valid_trial=np.zeros([1,subj_count],dtype=int)
	error_distances = np.zeros([300,subj_count],dtype=float)
	subj_index=-1
	feedbackCount=0
	firstTime=True
	correct_loc = np.zeros([subj_count,1],dtype=int)
	total_loc = np.zeros([subj_count,1],dtype=int)
	for subjects in subjects_id:
		total_sess=get_total_sess_count(subjects)
		# print("total sessions: " + str(total_sess))
		print("for subject" + str(subjects))
		#update the subj_index
		subj_index+=1
		feedbackCount=0
		for i in range(0,total_sess-1):
			logfile=data_dir+"sub"+subjects+"/session_"+str(i)+"/Beh/sub"+subjects+"Log.txt"
			print(logfile)
			if os.path.isfile(logfile):
				inFile = open(logfile)
				items=np.zeros([40,6],dtype='|S30')
				trialindex=-1
				itemindex=0
				leftitem=""
				rightitem=""
				locChoice=False
				index=0
				correct_response=False
				pos=-1 #0 is left,1  is right
				called=0
				added=0
				feedbackIndex=0

				posX=0
				posZ=0
				correctX=0
				correctZ=0
				
				check=False
				checkindex=0
				rightname=""
				for s in inFile.readlines():
					s = s.replace('\r','8')
					tokens = s[:-1].split('\t')
					if len(tokens)>1:
						phase=-999
						if tokens[3]=="TREASURE_LABEL":
							# print(itemindex)
							items[trialindex,itemindex]=str(tokens[4])
							# print("adding to items " + str(tokens[4]))
							itemindex+=1
						if tokens[2]=="Trial Event":
							# print(tokens[3])
							if tokens[3]=="TRIAL_NAVIGATION_ENDED":
								itemindex=0
								trialindex+=1
								last_valid_trial[0,subj_index]=max(last_valid_trial[0,subj_index],trialindex)
							if tokens[3]=="LOCATION_RECALL_CHOICE_STARTED":
								locChoice=True
								feedbackIndex=-1
							if tokens[3]=="FEEDBACK_ENDED" and locChoice:
								locChoice=False
								feedbackIndex=-1
						if locChoice:
							tempFBIndex=feedbackIndex+1
							if tokens[2]=="coconut00"+str(tempFBIndex):
								if tokens[3]=="SPAWNED":
									feedbackIndex+=1
							if tokens[2]=="CorrectObjectIndicator00"+str(feedbackIndex):
								if tokens[3]=="POSITION":
									correctX = float(tokens[4])
									correctZ= float(tokens[6])
							if tokens[2]=="PositionSelectorVisuals00"+str(feedbackIndex):
								if tokens[3]=="POSITION":
									if firstTime:
										firstTime=False
									else:
										posX = float(tokens[4])
										posZ= float(tokens[6])
										error_distances[feedbackCount,subj_index] = ((correctX - posX)**2 + (correctZ-posZ)**2)
										# print("error dist is  " + str(error_distances[feedbackCount,subj_index]))
										feedbackCount+=1
										firstTime=True
							if tokens[2]=="PositionSelectorCenterSphere00"+str(feedbackIndex):
								if tokens[3]=="OBJECT_COLOR":
									total_loc[subj_index,0]+=1
									if float(tokens[4])==0.1764706:
										correct_loc[subj_index,0]+=1

	return [error_distances,correct_loc,total_loc]


								


def calculate_temporal_perf(data_dir,subjects_id):
	subj_count = len(subjects_id)
	print("subjects count is: " + str(subj_count))
	total=np.zeros([40,subj_count],dtype=int)
	perc=np.zeros([40,subj_count],dtype=float)
	correct_index=np.zeros([5,subj_count],dtype=int)
	total_index=np.zeros([5,subj_count],dtype=int)    
	correct=np.zeros([40,subj_count],dtype=int)
	perc_index=np.zeros([5,subj_count],dtype=float)
	last_valid_trial=np.zeros([1,subj_count],dtype=int)
	subj_index=-1
	for subjects in subjects_id:
		total_sess=get_total_sess_count(subjects)
		# print("total sessions: " + str(total_sess))
		print("for subject" + str(subjects))
		#update the subj_index
		subj_index+=1
		for i in range(0,total_sess-1):
			logfile=data_dir+"sub"+subjects+"/session_"+str(i)+"/Beh/sub"+subjects+"Log.txt"
			print(logfile)
			if os.path.isfile(logfile):
				inFile = open(logfile)
				items=np.zeros([40,6],dtype='|S30')
				trialindex=-1
				itemindex=0
				leftitem=""
				rightitem=""
				tempfeedback=False
				index=0
				correct_response=False
				pos=-1 #0 is left,1  is right
				called=0
				added=0
				check=False
				checkindex=0
				rightname=""
				for s in inFile.readlines():
					s = s.replace('\r','8')
					tokens = s[:-1].split('\t')
					if len(tokens)>1:
						phase=-999
						if tokens[3]=="TREASURE_LABEL":
							# print(itemindex)
							items[trialindex,itemindex]=str(tokens[4])
							# print("adding to items " + str(tokens[4]))
							itemindex+=1
						if tokens[2]=="Trial Event":
							# print(tokens[3])
							if tokens[3]=="TRIAL_NAVIGATION_ENDED":
								itemindex=0
								trialindex+=1
								last_valid_trial[0,subj_index]=max(last_valid_trial[0,subj_index],trialindex)
							if tokens[3]=="TEMPORAL_FEEDBACK_STARTED":
								tempfeedback=True
							if tokens[3]=="TEMPORAL_FEEDBACK_ENDED":
								tempfeedback=False
						if tokens[3]=="SCORE_ADDED_SEQUENCE":
							correct_response=True
						if check==True:
							if tokens[2]=="Object B Name Text":
								"nothing"
							else:
								if rightitem not in "":
									called+=1
									compare_items(leftitem,rightitem,items[trialindex-1,0:4],trialindex,correct_response,subj_index,correct_index,correct,total_index,total)
									correct_response=False
							check=False
						if tokens[2]=="Object A Name Text" and tempfeedback:
							if tokens[3]=="TEXT_MESH":
								leftitem=tokens[4]
								check=True
								rightname=rightitem
								# print("left item " + str(leftitem))
						if tokens[2]=="Object B Name Text" and tempfeedback:
							if tokens[3]=="TEXT_MESH":
								rightitem=tokens[4]
								called+=1
								# print("right item " + str(rightitem))
								compare_items(leftitem,rightitem,items[trialindex-1,0:4],trialindex,correct_response,subj_index,correct_index,correct,total_index,total)
								correct_response=False
	subj_count = len(subjects_id)
	perc_mean=np.zeros([subj_count,1],dtype=float)
	print(np.shape(perc_mean))
	for j in range(0,subj_count):
		for i in range(0,last_valid_trial[0,j]):
			# print("last valid trial " + str(last_valid_trial[0,j]))
			if total[i,j]!=0:
				# print("perc: " + str(correct[i,j]) + " / " +  str(total[i,j]))
				perc[i,j]=float(correct[i,j])/float(total[i,j])
				# print(perc[i,j])
		perc_mean[j,0]=np.mean(perc[0:last_valid_trial[0,j],j]*100)
		print("perc mean is " + str(perc_mean[j,0]) + " for " + str(subjects_id[j]) + " with last valid trial " + str(last_valid_trial[0,j]))
		for i in range(0,5):   
			print(i)
			print(total_index)
			if total_index[i,j]!=0:
				perc_index[i,j]=float(correct_index[i,j])/(float(total_index[i,j]))
				print("perc index at :" + str(i) + " is " + str(perc_index[i,j]))
	return [perc_mean,perc_index]