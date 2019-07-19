import numpy as np
import sys
import os.path
#data_dir = "/Users/anshpatel/Desktop/Freiburg_analysis/behavioral/"
data_dir = "/home1/melina.tsitsiklis/eyetracking/behavioral"

#subjects_id = ["200218","150318","100418","240418"]
subjects_id = ["Melina"]

min_sacc_thresh=12
max_sacc_thresh=40

def calculate_vel(leftposX,leftposY,n,unique_samples,times):
    if n-2<=0 or n+2>=unique_samples:
        return 0
    else:
        time_diff=abs(times[n-2]-times[n+2])
        if time_diff>0: #nonzero time difference so velocity not infinite (only want unique samples)
            mov_vel=(leftposX[n-2,0]+leftposX[n-1,0]+leftposX[n+1,0]+leftposX[n+2,0])/(6*time_diff)
            return mov_vel
        else:
            return 0

def calculate_saccades(velocity,thresh,unique_samples,times):
	total=0
	zeroes_count=0
	mindistance=100
	current_dist=100
	prev_index=-1
	saccade_samples=np.zeros([1,1],dtype=int)
	for i in range(0,unique_samples-1): #get indices of samples that pass velocity threshold condition
		if velocity[i,0]>(6*thresh): #is vel more than 6x std dev per staudigl 2016
			saccade_samples=np.append(saccade_samples,i)
			current_dist=i-prev_index
			if current_dist<mindistance:
				mindistance=current_dist
			prev_index=i
			total+=1

	#calculate successive time differences
	succ_time_diff=np.zeros([len(saccade_samples)-1,1],dtype=int)
	window_sacc=0
	start_times=np.zeros([len(saccade_samples)-1,1],dtype=int)
	end_times=np.zeros([len(saccade_samples)-1,1],dtype=int)
	end_times_sacc = np.zeros([len(saccade_samples)-1,1],dtype=int)
	start_times_sacc = np.zeros([len(saccade_samples)-1,1],dtype=int)
	for i in range(0,len(saccade_samples)-1): #find samples that exceed 12ms min condition
		time_diff=times[saccade_samples[i+1,]]-times[saccade_samples[i,]]
		norm_time_diff=time_diff/1000 #unix time is orginially in microseconds
		succ_time_diff[i,0]=norm_time_diff
		if norm_time_diff==0:
			zeroes_count+=1
		if norm_time_diff<12: #want greater than 12ms
			increment_index=1
			temp_time_diff=norm_time_diff
			final_time_diff=0
			prev_time_diff=0
			succ_time_diff[i,0]=norm_time_diff
			while temp_time_diff < 40: #and also less than 40ms
				prev_time_diff=temp_time_diff[0,] 
				if i+increment_index < (len(saccade_samples)-1):
					temp_time_diff+=(times[saccade_samples[i+increment_index,]]-times[saccade_samples[i+(increment_index-1),]])/1000
				else:
					break
				increment_index+=1
				continue
			succ_time_diff[i,0]=prev_time_diff
			start_times[i,0]=times[saccade_samples[i,]]
			end_times[i,0]=times[saccade_samples[i+(increment_index-1)],]
			i+=(increment_index-1)
	[sacc_count, start_sacc_times,end_sacc_times] = return_saccade_count(succ_time_diff,saccade_samples,start_times,end_times)
	return [sacc_count,start_sacc_times,end_sacc_times]


def calculate_binocular_saccades(left_start_time,left_end_time,right_start_time,right_end_time,sacc_l,sacc_r):
	bino_sacc_count=0
	min_sacc_iter=0
	if sacc_l > sacc_r:
		min_sacc_iter=sacc_r
	else:
		min_sacc_iter=sacc_l
	start_times = np.zeros([min_sacc_iter,1],dtype=int)
	end_times = np.zeros([min_sacc_iter,1],dtype=int)
	for i in range(0,min_sacc_iter):
		l1 = left_start_time[i,0]
		l2 = left_end_time[i,0]
		for j in range(i,sacc_r):
			r1 = right_start_time[j,0]
			r2= right_end_time [j,0]
			if r2 > l1 and r1 < l2:
				# print r1
				bino_sacc_count+=1
				start_times[bino_sacc_count-1,0]=r1
				end_times[bino_sacc_count-1,0]=r2
				break
	start_times=np.trim_zeros(start_times)
	end_times=np.trim_zeros(end_times)
	return [bino_sacc_count,start_times,end_times]



def calculate_saccade_duration(veldata,start_frame,vel_thresh):
    append_index=0
    while veldata[start_frame+append_index,0]>(6*vel_thresh):
        append_index+=1
        continue
    return append_index

def return_saccade_count(succ_time_diff,saccade_samples,start_times,end_times):
	sacc_win=0
	start_sacc_times=np.zeros([len(saccade_samples)-1,1],dtype=int)
	end_sacc_times=np.zeros([len(saccade_samples)-1,1],dtype=int)
	for i in range(0,len(saccade_samples)-1):
		if succ_time_diff[i,0]>min_sacc_thresh and succ_time_diff[i,0]<max_sacc_thresh:
			end_sacc_times[sacc_win,0]=end_times[i,0]
			start_sacc_times[sacc_win,0]=start_times[i,0]
			sacc_win+=1
	return [sacc_win,start_sacc_times,end_sacc_times]

def return_trial_saccade_distribution(start_times,startTrialTime,endTrialTime):
	total=np.zeros([40,1],dtype=int)
	for j in range(0,len(start_times)):
		for i in range(0,len(startTrialTime)-1):
			if start_times[j,0] > startTrialTime[i,0] and start_times[j,0] < endTrialTime[i,0]:
				total[i,0]+=1
	return total


def get_total_sess_count(subjects):
    sess_count=0
    for i in range(0,3):
        logfolder=data_dir+"sub"+subjects+"/session_"+str(i)+"/"
        if os.path.isdir(logfolder):
            sess_count+=1
    return sess_count+1