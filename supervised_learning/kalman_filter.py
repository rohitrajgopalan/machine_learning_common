import math
import numpy as np
import pandas as pd
from harmony_ml import *


class KalmanFilter:
    #Store lat,lon,angle,velocity,decision
    historical_data = None
    earth_radius = 63178137.0

    #Store decision,number of records, counter, lat variance, lon variance, angle variance, velocity variance
    summarised_data = None

    nodeID = 0

    require_update = False

    P = np.eye(4)
    I = np.eye(4)
    R = np.eye(2)
    Q = np.diag([(0.5*8.8**2)**2,(0.5*8.8**2)**2,4.0,35.0**2])
    H = np.matrix([[1.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0]])
    decisions = []

    def __init__(self,nodeID, upload_data_from_csv=False):
        self.nodeID = nodeID
        self.historical_data = pd.DataFrame(columns = ['LAT','LON','ANGLE','VELOCITY','DECISION'])
        if upload_data_from_csv:
            node_data = get_data_for_nodeID(self.nodeID,0)
            node_data = node_data.drop(0)
            self.historical_data['LAT'] = node_data['NODE{0}LAT'.format(nodeID)]
            self.historical_data['LON'] = node_data['NODE{0}LON'.format(nodeID)]
            self.historical_data['ANGLE'] = node_data['NODE{0}DIRECTION_OF_TRAVEL'.format(nodeID)]
            self.historical_data['VELOCITY'] = node_data['NODE{0}DISTANCE_TRAVELLED'.format(nodeID)] * 3.6
            self.historical_data['DECISION'] = node_data['NODE{0}DECISION'.format(nodeID)]
            self.update_filter()

    def update_filter(self):
        self.decisions = list(np.unique(self.historical_data['DECISION']))
        variances = self.historical_data.groupby('DECISION')[['LAT','LON','ANGLE','VELOCITY']].std()

        data = []
        for decision in self.decisions:
            variance_row = variances.iloc[self.decisions.index(decision)]
            lat = variance_row['LAT']
            lon = variance_row['LON']
            angle = math.radians(variance_row['ANGLE'])
            velocity = variance_row['VELOCITY']/3.6
            X,Y = self.convert_to_X_Y_coords(lat,lon)
            data.append([decision,Y,X,angle,velocity])

        self.summarised_data = pd.DataFrame(data,columns = ['DECISION','LAT','LON','ANGLE','VELOCITY'])
            
        
    def perform_filter(self,decision,lat,lon,angle,velocity):
        self.update_filter()
        
        if self.historical_data.loc[self.historical_data['DECISION'] == decision].shape[0] == 0 or self.summarised_data.loc[self.summarised_data['DECISION'] == decision].shape[0] == 0:
            self.historical_data = self.historical_data.append({'DECISION': decision, 'LAT':lat,'LON':lon,'ANGLE':angle,'VELOCITY':velocity},ignore_index=True)
            return lat,lon,angle,velocity
        else:
            summary_row = self.summarised_data.iloc[self.decisions.index(decision)]
            x = self.convert_to_X_Y_coords_from_matrix(np.matrix([[lon,lat,angle,velocity]]).T)
            a13 = x[3,0]*np.sin(x[2,0])
            a14 = np.cos(x[2,0])
            a23 = x[3,0]*np.cos(x[2,0])
            a24 = np.sin(x[2,0])
            A = np.matrix([[1.0, 0.0, a13, a14],
                    [0.0, 1.0, a23, a24],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
                            
            self.R = np.diag([summary_row['LON']**2,summary_row['LAT']**2])
            self.P = np.diag([summary_row['LON']**2,summary_row['LAT']**2,summary_row['ANGLE']**2,summary_row['VELOCITY']**2])

            self.P = A*self.P*A.T + self.Q
            S = self.H*self.P*self.H.T + self.R
            K = (self.P*self.H.T) * np.linalg.pinv(S)

            past_row = self.historical_data[self.historical_data['DECISION'] == decision].iloc[0]
            past_lat = past_row['LAT']
            past_lon = past_row['LON']

            z_x,z_y = self.convert_to_X_Y_coords(past_lat,past_lon)
            Z = np.matrix([[z_x,z_y]]).T

            hx = np.matrix([[float(x[0,0])],
                            [float(x[1,0])]])
            y = Z - (hx)
            x = x + (K*y)
            x = self.convert_from_X_Y_coords(x)
            self.P = (self.I - (K*self.H))*self.P

            new_lat = x[1,0]
            new_lon = x[0,0]
            new_angle = x[2,0]
            new_velocity = x[3,0]

            self.historical_data = self.historical_data.drop(self.historical_data[self.historical_data['DECISION'] == decision].index[0])
            self.historical_data = self.historical_data.append({'DECISION': decision, 'LAT':new_lat,'LON':new_lon,'ANGLE':new_angle,'VELOCITY':new_velocity},ignore_index=True)
            return new_lat,new_lon,new_angle,new_velocity

    def convert_to_X_Y_coords(self,lat,lon,angle=None,velocity=None):

        lam_rad = math.radians(lat)
        phi_rad = math.radians(lon)

        Y = self.earth_radius*lam_rad*np.cos(phi_rad)
        X = self.earth_radius*phi_rad

        if not angle is None and not velocity is None:
            Y += velocity * np.sin(angle)
            X += velocity * np.cos(angle)

        return X,Y

    def convert_to_X_Y_coords_from_matrix(self,x):
        x[2,0] = math.radians(x[2,0])
        x[3,0] = x[3,0]/3.6
        X,Y = self.convert_to_X_Y_coords(x[1,0],x[0,0],x[2,0],x[3,0])

        x[0,0] = X
        x[1,0] = Y    
        x[2,0] = (x[2,0]+np.pi) % (2.0*np.pi)- np.pi

        return x

    def convert_from_X_Y_coords(self,x):
        X = x[0,0]
        Y = x[1,0]

        phi_rad= X/self.earth_radius
        lam_rad = Y/(self.earth_radius*np.cos(phi_rad))

        x[0,0] = math.degrees(phi_rad)
        x[1,0] = math.degrees(lam_rad)
        x[2,0] = math.degrees(x[2,0])

        if x[2,0] < 0:
            x[2,0] += 360.0

        x[3,0] *= 3.6
        
        return x
