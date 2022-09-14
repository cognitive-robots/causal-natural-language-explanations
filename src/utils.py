# |**********************************************************************;
# Project           : Why Do We Stop? Textual Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

import  csv
import sys
from    sys             import platform
import  math
import  numpy           as np
from    scipy           import interpolate
import  cv2
import  os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
import matplotlib.transforms as trf
import seaborn as sb
from PIL import Image
import scipy
import io

class bcolors:
    HEADER  = '\033[95m'
    BLUE    = '\033[94m'
    GREEN   = '\033[92m'
    WARNING = '\033[93m'
    FAIL    = '\033[91m'
    ENDC    = '\033[0m'
    BOLD    = '\033[1m'
    HIGHL   = '\x1b[6;30;42m'
    UNDERLINE = '\033[4m'

class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def check_and_make_folder(path):
    if not os.path.exists(path): 
        os.makedirs(path)
        print(bcolors.BLUE+"Make a folder: {}".format(path)+bcolors.ENDC)

def csv_dict_reader(file_obj):
    """
    Read a CSV file using csv.DictReader
    """
    return csv.DictReader(file_obj, delimiter=',')

def get_vid_info(file_obj):
    if platform == 'darwin':
        nFrames     = int(file_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        img_width   = int(file_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height  = int(file_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = file_obj.get(cv2.CAP_PROP_FPS)
    else:
        nFrames     = int(file_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        img_width   = int(file_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height  = int(file_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = file_obj.get(cv2.CAP_PROP_FPS)

        # if you have a trouble with these codes,
        # try:
        # pip uninstall opencv-python
        # sudo apt-get install python-opencv
        # OpenCV3 might have an issue with loading a video

    return nFrames, img_width, img_height, fps

def refine_records(timestamp, record):
    timestamp   = np.array(timestamp)

    record_interp  = interpolate.interp1d(timestamp, record)
    idxs           = np.linspace(timestamp[0], timestamp[-1], timestamp.shape[0]).astype("float")
    out            = record_interp(idxs)

    return out

def compute_curvature(gps_x, gps_y):
    dx  = np.gradient(gps_x)
    dy  = np.gradient(gps_y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    num         = np.multiply(dx,ddy) - np.multiply(ddx,dy)
    denom       = np.multiply(dx,dx) + np.multiply(dy,dy)
    denom       = np.sqrt(denom)
    denom       = np.multiply(np.multiply(denom,denom),denom)
    denom[np.where(denom<=0.001)] = 0.001
    curvature   = np.divide(num,denom)
    curvature[np.where(denom<=0.001)] = 0

    return curvature

def get_equally_spaced_points( gps_x, gps_y ):
    d = np.diff( np.concatenate(([gps_x], [gps_y])), axis=1 )
    dist_from_vertex_to_vertex = np.hypot(d[0], d[1])+0.001
    cumulative_dist_along_path = np.concatenate(([0], dist_from_vertex_to_vertex.cumsum(axis=0)))
    num_points = int(np.floor(cumulative_dist_along_path[-1]))
    dist_steps = np.concatenate((range(num_points+1), [cumulative_dist_along_path[-1]]))

    gps_interp = interpolate.interp1d(cumulative_dist_along_path, np.concatenate(([gps_x], [gps_y])))
    points     = gps_interp(dist_steps)

    return points, dist_steps, cumulative_dist_along_path

def lla2flat(lla, llo, psio, href):
    '''
    lla  -- array of geodetic coordinates 
            (latitude, longitude, and altitude), 
            in [degrees, degrees, meters]. 
 
            Latitude and longitude values can be any value. 
            However, latitude values of +90 and -90 may return 
            unexpected values because of singularity at the poles.
 
    llo  -- Reference location, in degrees, of latitude and 
            longitude, for the origin of the estimation and 
            the origin of the flat Earth coordinate system.
 
    psio -- Angular direction of flat Earth x-axis 
            (degrees clockwise from north), which is the angle 
            in degrees used for converting flat Earth x and y 
            coordinates to the North and East coordinates.
 
    href -- Reference height from the surface of the Earth to 
            the flat Earth frame with regard to the flat Earth 
            frame, in meters.
 
    usage: print(lla2flat((0.1, 44.95, 1000.0), (0.0, 45.0), 5.0, -100.0))
    '''
    R = 6378137.0               # Equator radius in meters
    f = 0.00335281066474748071  # 1/298.257223563, inverse flattening
 
    Lat_p = lla[0] * math.pi / 180.0  # from degrees to radians
    Lon_p = lla[1] * math.pi / 180.0  # from degrees to radians
    Alt_p = lla[2]  # meters
 
    # Reference location (lat, lon), from degrees to radians
    Lat_o = llo[0] * math.pi / 180.0
    Lon_o = llo[1] * math.pi / 180.0
     
    psio = psio * math.pi / 180.0  # from degrees to radians
 
    dLat = Lat_p - Lat_o
    dLon = Lon_p - Lon_o
 
    ff = (2.0 * f) - (f ** 2)  # Can be precomputed
 
    sinLat = math.sin(Lat_o)
 
    # Radius of curvature in the prime vertical
    Rn = R / math.sqrt(1 - (ff * (sinLat ** 2)))
 
    # Radius of curvature in the meridian
    Rm = Rn * ((1 - ff) / (1 - (ff * (sinLat ** 2))))
 
    dNorth = (dLat) / math.atan2(1, Rm)
    dEast = (dLon) / math.atan2(1, (Rn * math.cos(Lat_o)))
 
    # Rotate matrice clockwise
    Xp = (dNorth * math.cos(psio)) + (dEast * math.sin(psio))
    Yp = (-dNorth * math.sin(psio)) + (dEast * math.cos(psio))
    Zp = -Alt_p - href
 
    return Xp, -Yp, Zp

def get_goalDirection(dist_steps, points):
    goalDirection = np.zeros((len(dist_steps)))

    for idx in range(len(dist_steps)):
        u = np.zeros((3))
        v = np.zeros((3))

        if idx != len(dist_steps)-1:
            u[0] = points[0][-1]-points[0][idx]
            u[1] = points[1][-1]-points[1][idx]
            v[0] = points[0][idx+1]-points[0][idx]
            v[1] = points[1][idx+1]-points[1][idx]

        ThetaInDegrees = math.atan2(np.linalg.norm(np.cross(u,v)), np.dot(u,v))
        ThetaSign      = np.sum(np.sign(np.cross(u,v)))
        goalDirection[idx] = ThetaInDegrees*ThetaSign;

    return goalDirection

# given a series and alpha, return series of smoothed points
def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)

# refine course information
def preprocess_course(course_value, nImg):
    nRecords = course_value.shape[0]

    for idx in range(1,nRecords):
        if course_value[idx] - course_value[idx-1] > 180:
            course_value[idx:] -= 360
        elif course_value[idx] - course_value[idx-1] < -180:
            course_value[idx:] += 360

    # interpolation
    xaxis         = np.arange(0, nRecords)
    idxs          = np.linspace(0, nRecords-1, nImg).astype("float")  # approximate alignment
    course_interp = interpolate.interp1d(xaxis, course_value)
    course_value  = np.expand_dims(course_interp(idxs),1)
    
    # exponential smoothing in reverse order
    course_value_smooth = np.flip(exponential_smoothing(np.flip(course_value,0), 0.01),0)
    course_delta        = course_value-course_value_smooth

    return course_delta


def preprocess_course_SAX(course_value, nImg):
    nRecords = course_value.shape[0]

    for idx in range(1, nRecords):
        if course_value[idx] - course_value[idx - 1] > 180:
            course_value[idx:] -= 360
        elif course_value[idx] - course_value[idx - 1] < -180:
            course_value[idx:] += 360

    # SAX: Compute change of course (sample to 10Hz and then compute gradient) -> Instead of exponential smoothing
    course_deriv = np.gradient(course_value)  # Result is deg/100ms
    course_deg_sec = course_deriv * 10  # change: BDDX 1, SAX 10

    for idx in range(1, nRecords):
        if np.abs(course_deg_sec[idx]) > 90:
            course_deg_sec[idx] = course_deg_sec[idx - 1]

    # interpolation
    xaxis = np.arange(0, nRecords)
    idxs = np.linspace(0, nRecords - 1, nImg).astype("float")  # approximate alignment
    course_interp = interpolate.interp1d(xaxis, course_deg_sec)
    course_value = np.expand_dims(course_interp(idxs), 1)

    return course_value

# refine other log information
def preprocess_others(series, nImg):
    nRecords = series.shape[0]

    # interpolation
    xaxis         = np.arange(0, nRecords)
    idxs          = np.linspace(0, nRecords-1, nImg).astype("float")  # approximate alignment
    series_interp = interpolate.interp1d(xaxis, series)
    series        = np.expand_dims(series_interp(idxs),1)

    return series

def visualize_attnmap(alphas, img):
    img_att = np.zeros_like(img)

    # Colour Map
    N = 255
    cmap = plt.cm.Reds
    cmap._init()
    cmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)

    y, x = np.mgrid[0:img.shape[1], 0:img.shape[0]]

    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    cb = ax.contourf(x, y, alphas.reshape(x.shape[0], y.shape[1]), 15, cmap=cmap)
    plt.colorbar(cb)
    plt.show()

    return img_att

def visualize_attnmap_2(alphas, img_raw):
    #img_att = np.zeros_like(img_raw)

    img = Image.fromarray((img_raw).astype(np.uint8))
    alp_map = np.reshape(alphas, (12, 20))


    fig, ax = plt.subplots(figsize=(20, 12))
    hmax = sb.heatmap(alp_map, cmap="jet", vmin= 0.0005, vmax=0.004, #0..1 Reds
           linewidth=0, cbar_kws={"shrink": .8}, xticklabels=False, yticklabels=False, zorder = 2, alpha = 0.5) #

    hmax.imshow(img,
                aspect=hmax.get_aspect(),
                extent=hmax.get_xlim() + hmax.get_ylim(),
                zorder=1)

    #plt.savefig('/root/Workspace/explainable-deep-driving-master/attn_sample.png')

    io_buf = io.BytesIO()
    plt.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_att = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    plt.close()

    return img_att

def visualize_attnmap_2_SAX(alphas, img_raw):
    #img_att = np.zeros_like(img_raw)

    img = Image.fromarray((img_raw).astype(np.uint8))
    alp_map = np.reshape(alphas, (12, 20))


    fig, ax = plt.subplots(figsize=(20, 12))
    hmax = sb.heatmap(alp_map, cmap="jet", vmin= 0.00000001, vmax=0.0000001, #0..1 Reds 0.00001
           linewidth=0, cbar_kws={"shrink": .8}, xticklabels=False, yticklabels=False, zorder = 2, alpha = 0.5) #

    hmax.imshow(img,
                aspect=hmax.get_aspect(),
                extent=hmax.get_xlim() + hmax.get_ylim(),
                zorder=1)

    #plt.savefig('/root/Workspace/explainable-deep-driving-master/attn_sample.png')

    io_buf = io.BytesIO()
    plt.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_att = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    plt.close()

    return img_att




