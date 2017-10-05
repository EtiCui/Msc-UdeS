#!/usr/bin/python

import numpy as np
import pandas as pd
from glob import glob
from cluster_analysis import cluster
import matplotlib.pyplot as plt

def cluster_traj(first_frame=-20):
    # create a list of all the clusters filename
    complete_traj = glob("cluster.*.out.gz")
    # sort the list
    complete_traj.sort(key=lambda f: int(filter(str.isdigit, f)))
    # consider only the desired frame
    wanted_frames = complete_traj[first_frame:]
    # list for all the cluster size and average cluster size and hairpin
    cluster_all = []
    average_cluster_size = []
    average_hairpins = []
    # for each file in the trajectory
    for frame in wanted_frames:
        #open the file
        cluster_info = cluster(fname=frame)
        #calculate the occurrence of each size
        frame_cluster_occurrence = cluster_info.occurrence()
        cluster_frame = list(frame_cluster_occurrence)
        cluster_all = cluster_all + cluster_frame
        #calulate the histogram for average size
        hist_frame,bins_frame,average_cluster_frame = cluster_info.histogram(frame_cluster_occurrence)
        average_cluster_size = average_cluster_size + [average_cluster_frame]
        # hairpins defects
        average_hairpins = average_hairpins + [cluster_info.hairpin()]

    # convert to array
    cluster_all = np.array(cluster_all)
    average_cluster_size = np.array(average_cluster_size)
    average_hairpins = np.array(average_hairpins)
    print("Average size and std: ",average_cluster_size.mean(),average_cluster_size.std())
    print("Average number of hairpin defect per chain and std: ",average_hairpins.mean(),average_hairpins.std())
    # create the histogram
    cluster_histogram, bins,useless = cluster_info.histogram(cluster_all)
    # save the file
    np.savetxt("cluster_histogram.out",
               np.transpose([bins, bins*cluster_histogram]))
    # matplotlib graph
    plt.xlabel("Taille")
    plt.ylabel("Nombre de particules")
    plt.bar(bins, bins * cluster_histogram)
    plt.show()
cluster_traj()
