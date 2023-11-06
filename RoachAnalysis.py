import h5py
import numpy as np
import os 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm

#dir = file path
# file = (r'C:\Users\verpeutlab\Documents\CockroachSLEAPDoNotDeletePlease\PredictionsAsH5\D11_Con_6.mp4.predictions.h5')
# file = (r'C:\Users\verpeutlab\Documents\CockroachSLEAPDoNotDeletePlease\PredictionsAsH5\D6_24_2.mp4.predictions.h5')
# file = (r'C:\Users\verpeutlab\Documents\CockroachSLEAPDoNotDeletePlease\PredictionsAsH5\D12_0_4.mp4.predictions.h5')
file = (r'C:\Users\verpeutlab\Documents\CockroachSLEAPDoNotDeletePlease\PredictionsAsH5\IMG_4483_2.mp4.predictions.h5')
#file = name of particular h5 file


with h5py.File(file, "r") as f:
    occupancy_matrix = f['track_occupancy'][:]
    tracks_matrix = f['tracks'][:]
    track_names = f['track_names'][:]

print(occupancy_matrix.shape)
print(tracks_matrix.shape)

# Generate basic data of the HDF5 file
filename = file
with h5py.File(filename, "r") as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

print("===filename===")
print(filename)
print()

print("===HDF5 datasets===")
print(dset_names)
print()

print("===locations data shape===")
print(locations.shape)
print()

print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")
print()

frame_count, node_count, _, instance_count = locations.shape

print("frame count:", frame_count)
print("node count:", node_count)
print("instance count:", instance_count)

#
from scipy.interpolate import interp1d

def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

locations = fill_missing(locations)
#
Head = 0
Metathorax = 1
Anus = 2
RFleg = 3
LFleg = 4
RHleg = 5
LHleg = 6 
RAntenna_tip = 7
LAntenna_tip = 8
LMleg = 9
RHleg_joint = 10
LHleg_joint = 11
Abdomen = 12
RMleg_joint = 13
LMleg_joint = 14
RAntenna_base = 15 
LAntenna_base = 16
Prothorax = 17
LFleg_joint = 18
RFleg_joint = 19
RMLeg = 20
RAntenna_mid = 21
LAntenna_mid = 22
RPalp_base = 23
LPalp_base = 24
RPalp_tip = 25
LPalp_tip = 26

Head_loc = locations[:, Head, :, :]
Metathorax_loc = locations[:, Metathorax, :, :]
Anus_loc = locations[:, Anus, :, :]
RFleg_loc = locations[:, RFleg, :, :]
LFleg_loc = locations[:, LFleg, :, :]
RHleg_loc = locations[:, RHleg, :, :]
LHleg_loc = locations[:, LHleg, :, :]
RAntenna_tip_loc = locations[:, RAntenna_tip, :, :]
LAntenna_tip_loc = locations[:, LAntenna_tip, :, :]
LMleg_loc = locations[:, LMleg, :, :]
RHleg_joint_loc = locations[:, RHleg_joint, :, :]
LHleg_joint_loc = locations[:, LHleg_joint, :, :]
Abdomen_loc = locations[:, Abdomen, :, :]
RMleg_joint_loc = locations[:, RMleg_joint, :, :]
LMleg_joint_loc = locations[:, LMleg_joint, :, :]
RAntenna_base_loc = locations[:, RAntenna_base, :, :]
LAntenna_base_loc = locations[:, LAntenna_base, :, :]
Prothorax_loc = locations[:, Prothorax, :, :]
LFleg_joint_loc = locations[:, LFleg_joint, :, :]
RFleg_joint_loc = locations[:, RFleg_joint, :, :]
RMLeg_loc = locations[:, RMLeg, :, :]
RAntenna_mid_loc = locations[:, RAntenna_mid, :, :]
LAntenna_mid_loc = locations[:, LAntenna_mid, :, :]
RPalp_base_loc = locations[:, RPalp_base, :, :]
LPalp_base_loc = locations[:, LPalp_base, :, :]
RPalp_tip_loc = locations[:, RPalp_tip, :, :]
LPalp_tip_loc = locations[:, LPalp_tip, :, :]

anchor = Head_loc
anchorName = 'Head'

videoSizeX = 300
videoSizeY = 1458
quarterLineBottom = 435 #Pixel on y-axis marking the 1st quarter
halfLine = 755 #Pixel on y-axis marking the 2nd quarter (or halfway)
quarterLineTop = 1075 #Pixel on y-axis marking the 3rd quarter


# tracks for entire video

sns.set('notebook', 'ticks', font_scale=1.2)

#To track for first x frames, uncomment below two lines below
#for i in range(1): #if multiple mice change 1 to # of mice
#    plt.plot(tracks_matrix[i,0,3,0:1000],tracks_matrix[i,1,3,0:1000]) #Replace 1000 with the number of frames desired

plt.plot(anchor[:,0,0],-1* anchor[:,1,0] + videoSizeY, 'k',label='Nicotine Roach 3')
# plt.plot(anchor[:,0,1],anchor[:,1,1], 'k',label='Roach 2')
#uncomment above to track multiple roaches

plt.legend()
leg = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=1)

plt.xlim(0,videoSizeX)
#plt.xticks([])

plt.ylim(0,videoSizeY)
#plt.yticks([])
plt.gca().set_aspect('equal', adjustable='box')
plt.title(anchorName+' tracks')
plt.gca().set_aspect('equal', adjustable='box')

# node tracks and velocity
# Determine time spent and distance traveled in total

fps = 30 #Frames per second of video
ppd = 20.76 #Pixel per distance; in this case, every 20.76 pixels equals 1cm

#Create a restricted area of interest by limiting x and y pixels of the video
box = pd.DataFrame({'x': anchor[:, 0,0], 'y': -1*anchor[:, 1,0] + videoSizeY }) #convert location tracks to dataframe. Metathorax chosen here.
boxRestrictX=box[(box['x'] >= 0 ) & (box['x'] <= videoSizeX)] #define location of restricted region of interest along x axis.
boxRestrictXY=box[(box['y'] >= 0) & (box['y'] <= videoSizeY)] #define location of restricted region of interest along y axis. In this case, it is the whole box

#total distance traveled
test=boxRestrictXY.to_numpy() #change back to array for distance calculation

totaldist = 0 #creates dataframe
for i in range(0, len(test)-1):
    totaldist += distance.euclidean(test[i], test[i+1])*(1/ppd)*(1/100) #calculates pixels per distance (in cm) and divides by 100 to obtain distance in meters

#total time in chamber
totaltime = box.shape[0]*(1/fps) #calculates time based on frames per second

plt.scatter(box['x'],box['y'], s = 0.1)

plt.xlim(0,videoSizeX)
#plt.xticks([])
plt.ylim(0,videoSizeY)
#plt.yticks([])
plt.gca().set_aspect('equal', adjustable='box')
plt.title(anchorName + ' tracks')


# 1st Quartile tracks
ROIX = box[(box['x'] >= -100) & (box['x'] <= videoSizeX)] # Define region of interest (ROI) along x axis for this quartile
Q1=box[(ROIX['y'] >= 0) & (ROIX['y'] < quarterLineBottom)] #Define region of interest along y axis for this quartile

#Quartile 1 distance traveled
Q1values=Q1.to_numpy()

Q1Distance = 0
for i in range(0, len(Q1values)-1):
    Q1Distance += distance.euclidean(Q1values[i], Q1values[i+1])*(1/ppd)*(1/100)

#time in Q1
Q1Time = Q1values.shape[0]*(1/fps)

# 2nd Quartile tracks
ROIX = box[(box['x'] >= -100) & (box['x'] <= videoSizeX)] # Define region of interest (ROI) along x axis for this quartile

print("Shape of box:", box.shape)
print("Index of box:", box.index)
print("Shape of ROIX:", ROIX.shape)
print("Index of ROIX:", ROIX.index)

Q2=box[(ROIX['y'] >= quarterLineBottom) & (ROIX['y'] < halfLine)] #Define region of interest along y axis for this quartile

#Quartile 2 distance traveled
Q2values=Q2.to_numpy()

Q2Distance = 0
for i in range(0, len(Q2values)-1):
    Q2Distance += distance.euclidean(Q2values[i], Q2values[i+1])*(1/ppd)*(1/100)

#time in Q2
Q2Time = Q2values.shape[0]*(1/fps)

# 3rd Quartile tracks
ROIX = box[(box['x'] >= -100) & (box['x'] <= videoSizeX)] # Define region of interest (ROI) along x axis for this quartile
Q3=box[(ROIX['y'] >= halfLine) & (ROIX['y'] < quarterLineTop)] #Define region of interest along y axis for this quartile

#Quartile 3 distance traveled
Q3values=Q3.to_numpy()

Q3Distance = 0
for i in range(0, len(Q3values)-1):
    Q3Distance += distance.euclidean(Q3values[i], Q3values[i+1])*(1/ppd)*(1/100)

#time in Q3
Q3Time = Q3values.shape[0]*(1/fps)

# 4th Quartile tracks
ROIX = box[(box['x'] >= -100) & (box['x'] <= videoSizeX)] # Define region of interest (ROI) along x axis for this quartile
Q4=box[( ROIX['y'] >= quarterLineTop ) & (ROIX['y'] < videoSizeY)] #Define region of interest along y axis for this quartile

#Quartile 4 distance traveled
Q4values=Q4.to_numpy()

Q4Distance = 0
for i in range(0, len(Q4values)-1):
    Q4Distance += distance.euclidean(Q4values[i], Q4values[i+1])*(1/ppd)*(1/100)

#time in Q4
Q4Time = Q4values.shape[0]*(1/fps)

#
print(file)
print('')
print('Total distance traveled (m):', totaldist)
print('Total time (s):', totaltime)
print('')
print('Quartile 1 distance traveled (m):', Q1Distance)
print('Time spent in Quartile 1 (s):', Q1Time) #seconds in defined component of the box
print('')
print('Quartile 2 distance traveled (m):', Q2Distance)
print('Time spent in Quartile 2 (s):', Q2Time) #seconds in defined component of the box
print('')
print('Quartile 3 distance traveled (m):', Q3Distance)
print('Time spent in Quartile 3 (s):', Q3Time) #seconds in defined component of the box
print('')
print('Quartile 4 distance traveled (m):', Q4Distance)
print('Time spent in Quartile 4 (s):', Q4Time) #seconds in defined component of the box
print('')

#%%Calculate boundary crossings

def count_boundary_crossings(y_coordinates, boundary_positions):
    crossings = [0] * len(boundary_positions)
    previous_position = y_coordinates[0]
    
    for position in y_coordinates[1:]:
        for i, boundary_y in enumerate(boundary_positions):
            if (previous_position < boundary_y and position >= boundary_y) or \
               (previous_position > boundary_y and position <= boundary_y):
                crossings[i] += 1
        previous_position = position
    
    return crossings

# Calculate boundary crossings for multiple boundaries for the 'Head' node
anchor_y_coordinates = -1*anchor[:, 1,0] + videoSizeY  
boundary_positions = [quarterLineBottom, halfLine, quarterLineTop]  # Specify the y-positions of the boundaries

crossings = count_boundary_crossings(anchor_y_coordinates, boundary_positions)

print("Number of crossings for each boundary:")
for i, boundary_y in enumerate(boundary_positions):
    print(f"Boundary {i + 1} ({boundary_y}): {crossings[i]} crossings.")
#%%Generate heatmap with scatter

#make heatmaps
x = box['x']
y = box['y']

fig, axs = plt.subplots(1, 2)

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

sigmas = [0, 64]
for ax, s in zip(axs.flatten(), sigmas):
    if s == 0:
        ax.plot(x, y, 'k.', markersize=1)
        ax.set_title("Scatter plot")
        ax.set_xlim(50,230)
        ax.set_ylim(0,1440)
    else:
        img, extent = myplot(x, y, s)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
        ax.set_xlim(0,videoSizeX)
        ax.set_ylim(0,videoSizeY)
        ax.set_title("Smoothing with  $sigma$ = %d" % s)
        # plt.savefig(file + "centroidscatterandheatmap.jpg")

plt.show()
#%%Generate line plot in different colors per chamber

#make figure for total chamber tracks
plt.plot(box['x'],box['y'],'grey')
plt.gca().set_aspect('equal', adjustable='box')
plt.title(anchorName  + ' tracks')

#make figure for quadrant 1 tracks
plt.plot(Q1['x'],Q1['y'],'r')
plt.gca().set_aspect('equal', adjustable='box')

#make figure for quadrant 2 tracks
plt.plot(Q2['x'],Q2['y'],'b')
plt.gca().set_aspect('equal', adjustable='box')

#make figure for quadrant 3 tracks
plt.plot(Q3['x'],Q3['y'],'g')
plt.gca().set_aspect('equal', adjustable='box')

#make figure for quadrant 4 tracks
plt.plot(Q4['x'],Q4['y'],'k')
plt.gca().set_aspect('equal', adjustable='box')

plt.xlim(0,videoSizeX)
plt.ylim(0,videoSizeY)

# plt.savefig(file + "centroidtracks.jpg")
#%% # Determine time spent and distance traveled around certain area (in this case, it is the nicotine pump)

#Quartile 1 distance traveled
Q1values=Q1.to_numpy()

Q1Distance = 0
for i in range(0, len(Q1values)-1):
    Q1Distance += distance.euclidean(Q1values[i], Q1values[i+1])*(1/ppd)*(1/100)

#time in Q1
Q1Time = Q1values.shape[0]*(1/fps)

#%% # Visualize x-y dynamics and velocity

from scipy.signal import savgol_filter

def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] array
    
    win defines the window to smooth over
    
    poly defines the order of the polynomial
    to fit with
    
    """
    node_loc_vel = np.zeros_like(node_loc)
    
    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)
    
    node_vel = np.linalg.norm(node_loc_vel,axis=1)

    return node_vel

centroid_vel_roach = smooth_diff(anchor[:, :, 0])

fig = plt.figure(figsize=(15,7))
ax1 = fig.add_subplot(211)
ax1.plot(anchor[:, 0, 0], 'r', label='x')
ax1.plot(-1*anchor[:, 1, 0] + videoSizeY, 'b', label='y')
ax1.legend()
ax1.set_title('Centroid')

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.imshow(centroid_vel_roach[:,np.newaxis].T, aspect='auto', vmin=0, vmax=4)
ax2.set_title('Velocity')

#%% # Visualize node colored by magnitude of speed

fig = plt.figure(figsize=(1,6))
kp = centroid_vel_roach  
vmin = 0
vmax = 4

fig = plt.scatter(anchor[:,0,0], videoSizeY - anchor[:,1,0], c=kp, s=2, vmin=vmin, vmax=vmax)
plt.set_xlim(0,videoSizeX)
plt.set_ylim(0,videoSizeY)
plt.set_title('Centroid tracks colored by magnitude of cockroach speed')

#%%
import pandas as pd
import cv2
import matplotlib.pyplot as plt

Video_FILE = r"C:\Users\verpeutlab\Documents\CockroachSLEAPDoNotDeletePlease\Poster Figures\IMG_4483_2.mov"

def get_frames(filename):
    video=cv2.VideoCapture(filename)
    while video.isOpened():
        rete,frame=video.read()
        if rete:
            yield frame
        else:
            break
        video.release()
        yield None

def get_frame(filename,index):
    counter=0
    video=cv2.VideoCapture(filename)
    while video.isOpened():
        rete,frame=video.read()
        if rete:
            if counter==index:
                return frame
            counter +=1
        else:
            break
    video.release()
    return None

frame = get_frame(Video_FILE,100)
print('shape is', frame.shape)
print('pixel at (60,21)',frame[60,21,:])
print('pixel at (120,10)',frame[120,10,:])
for f in get_frames(Video_FILE):
    if f is None: break
    cv2.imshow('frame',f)
    if cv2.waitKey(10) == 40: 
        break
cv2.destroyAllWindows()

img = np.flipud(frame)
img = img/255.0

plt.figure(figsize=(15,15)) 

# Uncomment below to show tracks (50 is added to the y axid to account for video resizing)
# plt.plot(anchor[:,0,0],videoSizeY - anchor[:,1,0] + 50, 'k')

#Uncomment below to show tracks color-coded for each quadrant (50 is added to the y axid to account for video resizing)
plt.plot(Q1['x'],Q1['y']+50,'r')
plt.plot(Q2['x'],Q2['y']+50,'b')
plt.plot(Q3['x'],Q3['y']+50,'g')
plt.plot(Q4['x'],Q4['y']+50,'k')

# Uncomment below to show tracks as velocity heat map (50 is added to the y axid to account for video resizing)
# plt.scatter(anchor[:,0,0], videoSizeY - anchor[:,1,0] + 50, c=kp, s=2, vmin=vmin, vmax=vmax)

# Do not uncomment below; this controld figure aesthetics
plt.xticks(np.arange(0, len(anchor[:,0,0])+1, 20),fontsize=10, rotation=45)
plt.xlim(0,videoSizeX)
plt.yticks(np.arange(0, len(anchor[:,0,0])+1, 20))
plt.ylim(0,videoSizeY)
plt.gca().set_aspect('equal', adjustable='box')
plt.title(anchorName + ' tracks')
plt.imshow(img)


# plt.savefig(file + "movieandtracks.jpg")

#%% # Unsupervised clustering of behaviors using kmeans

from sklearn.cluster import KMeans

def instance_node_velocities(instance_idx):
    node_locations = locations[:, :, :, instance_idx]
    node_velocities = np.zeros((frame_count, node_count))

    for n in range(0, node_count):
        node_velocities[:, n] = smooth_diff(node_locations[:, n, :])
    
    return node_velocities

def plot_instance_node_velocities(instance_idx, node_velocities):
    plt.figure(figsize=(20,8))
    plt.imshow(node_velocities.T, aspect='auto', vmin=0, vmax=5, interpolation="nearest")
    plt.xlabel('frames')
    plt.ylabel('nodes')
    plt.yticks(np.arange(node_count), node_names, rotation=20);
    plt.title(f'Cockroach {roach_ID} node velocities')
    
roach_ID = 0
node_velocities = instance_node_velocities(roach_ID)
plot_instance_node_velocities(roach_ID, node_velocities)

nstates = 5
km = KMeans(n_clusters=nstates)

labels = km.fit_predict(node_velocities)

# Calculate total number of seconds per cluster
unique, counts = np.unique(labels, return_counts=True)
counts_sorted = np.sort(counts)
#print("counts_sorted", counts_sorted)
behaviors = labels.tolist() #time in a behavior in seconds
print ("Count for 0", behaviors.count(0)/fps)
print ("Count for 1", behaviors.count(1)/fps)
print ("Count for 2", behaviors.count(2)/fps)
print ("Count for 3", behaviors.count(3)/fps)
print ("Count for 4", behaviors.count(4)/fps)
#print ("Count for 5", behaviors.count(5)/fps)
#print ("Count for 6", behaviors.count(6)/fps)
#print ("Count for 7", behaviors.count(7)/fps)
#print ("Count for 8", behaviors.count(8)/fps)
#print ("Count for 9", behaviors.count(9)/fps)
#print ("Count for 10", behaviors.count(10)/fps)

fig = plt.figure(figsize=(20, 12))

ax1 = fig.add_subplot(211)
ax1.imshow(node_velocities.T, aspect="auto", vmin=0, vmax=5, interpolation="nearest")
ax1.set_xlabel("Frames")
ax1.set_ylabel("Nodes")
ax1.set_yticks(np.arange(node_count))
ax1.set_yticklabels(node_names);
ax1.set_title(f"Cockroach {roach_ID} node velocities")
ax1.set_xlim(0,frame_count)

ax2 = fig.add_subplot(212,sharex=ax1)
ax2.imshow(labels[None, :], aspect="auto", cmap="tab10", interpolation="nearest")
ax2.set_xlabel("Frames")
ax2.set_yticks([])
ax2.set_title("Ethogram (colors = clusters)");
