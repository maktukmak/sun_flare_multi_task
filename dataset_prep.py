import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
from datetime import timedelta, datetime, time
import pickle
from random import sample, randrange
from PIL import Image
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
from astropy.io import fits

class dataset_struct():
    def __init__(self, dataset_cfg):
        self.cfg = dataset_cfg
        self.path_goes = r'C:\datasets\SMARP\\GOES.CSV'
        self.path_cache = r'C:\datasets\cache\\'
        
    class dataset():
        def __init__(self):
            pass

    def load_datasets(self):
        if self.cfg.use_cached_harps:
            with open(self.path_cache + "cache_hmi.txt", "rb") as fp:
                self.dataset_hmi = pickle.load(fp) 
            with open(self.path_cache + "cache_mdi.txt", "rb") as fp:
                self.dataset_mdi = pickle.load(fp)
                
        else:
            self.dataset_hmi = self.dataset()
            
            self.dataset_hmi.path_harp =  r'Z:\data2\SHARP_720s\header_los\\' 
            self.dataset_hmi.features = ['USFLUX', 'MEANGBZ', 'R_VALUE', 'AREA']
            self.dataset_hmi.cadence = 12
            self.dataset_hmi.mean = 0
            self.dataset_hmi.std = 1
            
            self.dataset_mdi = self.dataset()
            self.dataset_mdi.path_harp =  r'Z:\data2\SMARP\header\\'
            self.dataset_mdi.features = ['USFLUX', 'MEANGBZ', 'R_VALUE', 'AREA']
            self.dataset_mdi.cadence = 96
            self.dataset_mdi.mean = 0
            self.dataset_mdi.std = 1
            
            
            
    def preprocess_datasets(self):
               
        if self.cfg.use_cached_preprocessing:
            with open(self.path_cache + "cache_hmi_prc.txt", "rb") as fp:
                self.dataset_hmi = pickle.load(fp) 
            with open(self.path_cache + "cache_mdi_prc.txt", "rb") as fp:
                self.dataset_mdi = pickle.load(fp) 
            
        else:
            self.goes = self.goes_read_process()
            
            if not self.cfg.use_cached_harps:
                self.cache_harps()
            
            self.valid_flare_events(self.dataset_hmi)
            self.valid_flare_events(self.dataset_mdi)
            
            if self.cfg.remove_C:
                self.remove_c_flares(self.dataset_hmi)
                self.remove_c_flares(self.dataset_mdi)
        
            self.train_test_split(self.dataset_hmi, self.cfg.hmi_split_opt)
            self.train_test_split(self.dataset_mdi, self.cfg.mdi_split_opt)
            
            self.compute_normalization_stats(self.dataset_hmi)
            self.compute_normalization_stats(self.dataset_mdi)
            
            self.find_intersection()
            self.valid_overlap()
        
            with open(self.path_cache + "cache_hmi_prc.txt", "wb") as fp:
                pickle.dump(self.dataset_hmi, fp)
                
            with open(self.path_cache + "cache_mdi_prc.txt", "wb") as fp:
                pickle.dump(self.dataset_mdi, fp)
                
    def goes_read_process(self):
    
        goes = pd.read_csv(self.path_goes, delimiter=',')
        goes['peak_time'] = pd.to_datetime(goes['peak_time'])
        goes = goes.dropna(subset=['goes_class'])
        goes = goes.loc[(goes['goes_class'].str.len() == 4)]
        goes = goes.drop(goes.iloc[np.where(goes['goes_class'].str[:1] == 'A')[0]].index)
        
        intensity_letter = goes['goes_class'].str[:1]
        intensity = np.zeros(len(intensity_letter))
        intensity[np.where(intensity_letter == 'B')[0]] = 1e-7
        intensity[np.where(intensity_letter == 'C')[0]] = 1e-6
        intensity[np.where(intensity_letter == 'M')[0]] = 1e-5
        intensity[np.where(intensity_letter == 'X')[0]] = 1e-4
        intensity = intensity * goes['goes_class'].str[1:].astype(float)
        intensity = np.log(intensity)
        goes['intensity'] = intensity
        
        return goes
    
    def valid_flare_events(self, dataset):

        dataset.valid_events = []
        dataset.harp_stats = []
        dataset.cnt_harp_noaa = 0
        dataset.cnt_harp_wo_flare = 0
        dataset.cnt_flare_m_sharp = 0
        for harp_file in dataset.harp_data.keys():
            
            harp, start_time, end_time, _ = self.read_harp(harp_file, dataset)
            
            dataset.harp_stats.append([harp_file, start_time, end_time, end_time - start_time])
            
            
            noaa = harp['NOAA_AR'].unique()
            
            if len(noaa) == 1 and noaa[0] != 0:  #!!!!!!!!!!!!!
                
                goes_vis = self.goes_filtering(noaa)
                
                classes = goes_vis[['goes_class', 'peak_time']]
                delta = timedelta(minutes=self.cfg.len_pred + self.cfg.len_seq)
                valid_flares = classes.iloc[np.where(classes['peak_time'] - start_time > delta)[0]]
                
                if len(valid_flares) > 0:
                    data = []
                    for i in range(len(valid_flares)):
                        start_wind = valid_flares.iloc[i].values[1] - timedelta(minutes=self.cfg.len_pred + self.cfg.len_seq)
                        end_wind = valid_flares.iloc[i].values[1] - timedelta(minutes=self.cfg.len_pred)
                        if ((harp['T_REC'] >= start_wind) & (harp['T_REC'] < end_wind)).sum() >= (self.cfg.len_seq / dataset.cadence) - (self.cfg.len_seq / dataset.cadence)*0.2:
                            harp_features = harp[dataset.features][(harp['T_REC'] >= start_wind) & (harp['T_REC'] < end_wind)]
                            if harp_features.isna().sum().sum() < (self.cfg.len_seq / dataset.cadence) * len(dataset.features) * 0.05:
                                data.append([harp_file,valid_flares.iloc[i].values[0], valid_flares.iloc[i].values[1]])
                        else:
                            dataset.cnt_flare_m_sharp += 1
                        
                    #data = [ [harp_file,valid_flares.iloc[i].values[0], valid_flares.iloc[i].values[1]] for i in range(len(valid_flares))]
                    dataset.valid_events = dataset.valid_events + data
                else:
                    dataset.cnt_harp_wo_flare += 1
            else:
                dataset.cnt_harp_noaa += 1
                
        dataset.harp_stats = pd.DataFrame(dataset.harp_stats)
        
    
    def train_test_split(self, dataset, split_opt):
        
        dataset.valid_events_train = []
        dataset.valid_events_test = []
        
        if split_opt == 'date':
            for harp_file, flare, time in dataset.valid_events:
                if time.year >= 2015:
                    dataset.valid_events_test.append([harp_file, flare, time])
                else:
                    dataset.valid_events_train.append([harp_file, flare, time])
        elif split_opt == 'random':
            I = len(dataset.valid_events)
            ind = np.arange(I)
            np.random.shuffle(ind)
            ind_train = ind[0:int(I*0.80)]
            ind_test = ind[int(I*0.80):]
            dataset.valid_events_train = [dataset.valid_events[i] for i in ind_train]
            dataset.valid_events_test = [dataset.valid_events[i] for i in ind_test]
        else:
            dataset.valid_events_train = dataset.valid_events[:]
            dataset.valid_events_test = []
    
    def compute_normalization_stats(self, dataset):
        
        train_features = []
        for data in dataset.valid_events_train:
            train_features.append(self.read_features(data, dataset)[0])
            
        train_features = np.concatenate(train_features, axis = 0)
        dataset.mean = np.nanmean(train_features, axis = 0)
        dataset.std = np.nanstd(train_features, axis = 0)
        
            
    def remove_c_flares(self, dataset):
        ind_B = [idx for idx, element in enumerate(dataset.valid_events) if element[1][:1] == 'B']
        ind_MX = [idx for idx, element in enumerate(dataset.valid_events) if element[1][:1] == 'M' or element[1][:1] == 'X']
        dataset.valid_events = [dataset.valid_events[i] for i in (ind_B + ind_MX)]
        # ind_B = [idx for idx, element in enumerate(dataset.valid_events_test) if element[1][:1] == 'B']
        # ind_MX = [idx for idx, element in enumerate(dataset.valid_events_test) if element[1][:1] == 'M' or element[1][:1] == 'X']
        # dataset.valid_events_test = [dataset.valid_events_test[i] for i in (ind_B + ind_MX)]
        
        
    def find_intersection(self):
        valid_events_train_intersect = []
        for file, flare, time in self.dataset_mdi.valid_events_train:
            for file2, flare2, time2 in self.dataset_hmi.valid_events_train:
                if [flare, time] ==  [flare2, time2]:
                    valid_events_train_intersect.append([file, file2, flare, time])
                    break
                
        for file, file2, flare, time in valid_events_train_intersect:
            self.dataset_mdi.valid_events_train.remove([file ,flare, time])
            self.dataset_hmi.valid_events_train.remove([file2 ,flare, time])
                    
        self.dataset_hmi.valid_events_train_intersect = valid_events_train_intersect
        self.dataset_mdi.valid_events_train_intersect = valid_events_train_intersect

    def cache_harps(self):
        
        self.dataset_hmi.harp_data = {}
        for harp_file in os.listdir(self.dataset_hmi.path_harp):
            harp = self.read_harp(harp_file, self.dataset_hmi, cache = False)[0]
            self.dataset_hmi.harp_data[harp_file] = harp
            
        self.dataset_mdi.harp_data = {}
        for harp_file in os.listdir(self.dataset_mdi.path_harp):
            harp = self.read_harp(harp_file, self.dataset_mdi, cache = False)[0]
            self.dataset_mdi.harp_data[harp_file] = harp
            
        with open(self.path_cache + "cache_hmi.txt", "wb") as fp:
            pickle.dump(self.dataset_hmi, fp)
            
        with open(self.path_cache + "cache_mdi.txt", "wb") as fp:
            pickle.dump(self.dataset_mdi, fp)
    

    def goes_filtering(self, noaa):
                   
        noaa = noaa[0]
            
        goes_vis = self.goes.loc[self.goes['noaa_active_region'] == noaa]
        if self.cfg.max_flare_filtering and len(goes_vis) > 0:
            goes_vis_filt = []
            delta = timedelta(hours=12)
            for i in range(len(goes_vis)):
                start = goes_vis.iloc[i]['peak_time'] - delta
                end = goes_vis.iloc[i]['peak_time'] + delta
                neighbors = goes_vis.loc[(goes_vis['peak_time'] >= start) & (goes_vis['peak_time'] < end)]
                if self.cfg.max_flare_window_drop:
                    neighbors = neighbors.drop(goes_vis.iloc[i].name)
                    if all(goes_vis.iloc[i]['intensity'] >= neighbors['intensity']):
                        goes_vis_filt.append(goes_vis.iloc[i])
                else:
                    if len(neighbors)>0:
                        goes_vis_filt.append(goes_vis.iloc[i])
                        #goes_vis_filt.append(neighbors.iloc[neighbors['intensity'].argmax()])
                        goes_vis_filt[-1][['goes_class', 'intensity']] = neighbors.iloc[neighbors['intensity'].argmax()][['goes_class', 'intensity']]
                    
            goes_vis = pd.DataFrame(goes_vis_filt)
                
        return goes_vis

    def valid_overlap(self):
        
        
        start_tarp = 13404
        end_harp = 225
        valid_overlap_data = []
        for i in range(len(self.dataset_mdi.harp_stats)):
            
            if int(self.dataset_mdi.harp_stats.iloc[i][0][4:10]) >= start_tarp:
                for j in range(len(self.dataset_hmi.harp_stats)):
                    if int(self.dataset_hmi.harp_stats.iloc[j][0][4:10]) <= end_harp:
                        s0 = self.dataset_mdi.harp_stats.iloc[i][1]
                        e0 = self.dataset_mdi.harp_stats.iloc[i][2]
                        
                        s1 = self.dataset_hmi.harp_stats.iloc[j][1]
                        e1 = self.dataset_hmi.harp_stats.iloc[j][2]
                        
                        i0 = pd.Interval(s0, e0)
                        i1 = pd.Interval(s1, e1)
                        
                        if i0.overlaps(i1):
                            s = max(s0, s1)
                            e = min(e0, e1)
                            
                            # Check MDI data for nan values
                            df_mdi = self.read_harp(self.dataset_mdi.harp_stats.iloc[i][0], self.dataset_mdi)[0]
                            df_mdi = df_mdi[(df_mdi['T_REC'] > s) &  (df_mdi['T_REC'] < e)]
                            indnan = df_mdi.isna().sum(axis = 1)
                            
                            
                            df_hmi = self.read_harp(self.dataset_hmi.harp_stats.iloc[j][0], self.dataset_hmi)[0]
                            df_mdi = df_mdi[(df_mdi['T_REC'] > s) &  (df_mdi['T_REC'] < e)]
                            
                            
                            prev = -1
                            tmp = []
                            cnt = 1
                            for k in range(len(indnan)):
                                if (indnan.iloc[k] == 0) and (prev != 0):
                                    st = k
                                    cnt = 1
                                elif (indnan.iloc[k] == 0) and (k != len(indnan)-1):
                                    cnt +=1
                                elif ((indnan.iloc[k] != 0) and (prev == 0)):
                                    tmp.append([st, cnt])
                                elif (indnan.iloc[k] == 0) and (k == len(indnan)-1):
                                    tmp.append([st, cnt])
                                prev = indnan.iloc[k]
                            
                            for block in tmp:
                                if block[1] > 10:
                                    s = df_mdi['T_REC'].iloc[block[0]]
                                    e = df_mdi['T_REC'].iloc[block[0] + block[1]]
                                    
                                    exp_mdi = ((e-s).total_seconds() / 60) / self.dataset_mdi.cadence
                                    exp_hmi = ((e-s).total_seconds() / 60) / self.dataset_hmi.cadence
                                    
                                    tot_hmi = len(df_hmi[(df_hmi['T_REC'] > s) &  (df_hmi['T_REC'] < e)])
                                    if (tot_hmi >= exp_hmi-2) and (block[1] > exp_mdi-2):
                                        valid_overlap_data.append([self.dataset_mdi.harp_stats.iloc[i][0], self.dataset_hmi.harp_stats.iloc[j][0], s, e])
        self.dataset_hmi.valid_overlap_data = valid_overlap_data
        self.dataset_mdi.valid_overlap_data = valid_overlap_data
        
        
    def sample_overlap_data(self):
    
        delta = timedelta(minutes=self.cfg.len_seq)
        delta2 = timedelta(minutes=self.cfg.len_pred)#+ self.cfg.len_pred)
        while True:
            data = sample(self.dataset_hmi.valid_overlap_data, 1)[0]
            if (data[3] - data[2]) >  delta:
                break 
            
        interval = (data[3] - data[2] - delta)
        int_delta = (interval.days * 24 * 60 * 60) + interval.seconds
        random_second = randrange(int_delta)
            
        return [data[0], data[1], 'N0.0', data[2] + delta + delta2 + timedelta(seconds=random_second)]
    
    def data_to_image_file(self, data, dataset):
        
        
        
        data_end_time = data[2] - timedelta(minutes=self.cfg.len_pred)
        
        t = data_end_time.to_pydatetime()
        mod_time = time(t.hour, t.minute//dataset.cadence*dataset.cadence)
        t = datetime.combine(t.date(), mod_time)
        
        file = 'hmi.sharp_cea_720s.'
        file += str(int(data[0][4:10]))
        file += '.'
        file += str(t.year)
        file += str(t.month).zfill(2)
        file += str(t.day).zfill(2)
        file += '_'
        file += str(t.hour).zfill(2)
        file += str(t.minute).zfill(2)
        file += '00_TAI.magnetogram.fits'
        
        return file
    
    def read_features(self, data, dataset):
        
        harp, _, _, video = self.read_harp(data[0], dataset, img_read = True)
        
        data_start_time = data[2] - timedelta(minutes=self.cfg.len_pred + self.cfg.len_seq)
        data_end_time = data[2] - timedelta(minutes=self.cfg.len_pred)
        
        harp_features = np.zeros((int(self.cfg.len_seq/dataset.cadence), len(dataset.features)))
        if dataset.par_on:
            harp_features = harp[dataset.features][(harp['T_REC'] >= data_start_time) & (harp['T_REC'] < data_end_time)].interpolate(method='linear', axis = 0).ffill().bfill().values
            harp_features = (harp_features - dataset.mean) / dataset.std
            harp_features = np.pad(harp_features, ((0, int(self.cfg.len_seq/dataset.cadence - harp_features.shape[0])), (0,0)), 'mean')
            harp_features = harp_features[0:int(self.cfg.len_seq/dataset.cadence),:]
        
        img_features = np.zeros((int(self.cfg.len_seq/dataset.cadence), 1, 1))

        return [harp_features, img_features]
    
    def read_harp(self, harp_file, dataset, cache = True, img_read = False):
            
        if cache == True:
            harp = dataset.harp_data[harp_file]
        else:               
            features = ['T_REC', 'NOAA_AR'] + dataset.features
            if dataset.img_on:
                features += ['frame']
            
            harp = pd.read_csv(dataset.path_harp + harp_file, delimiter=',', usecols = features)
            harp['T_REC'] = harp['T_REC'].str[:-4]
            harp['T_REC'] = harp['T_REC'].str.replace('_', ' ')
            harp['T_REC'] = pd.to_datetime(harp['T_REC'])
        start_time = harp['T_REC'].iloc[0]
        end_time = harp['T_REC'].iloc[-1]
        
        video = 0
        
        return harp, start_time, end_time, video
    