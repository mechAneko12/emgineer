import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import pickle
import json

class EmgDecomposition():
    def __init__(self, 
                 n_motor_unit: int,
                 n_delayed: int=8,
                 threshold_sil: float=0.8,
                 random_state: int=None,
                 max_iter: int=200,
                 tol: float=1e-4,
                 cashe: str or None=None,
                 flag_sil: bool=True,
                 flag_pca: bool=False):
        self.n_motor_unit = n_motor_unit
        self.n_delayed = n_delayed
        self.threshold_sil = threshold_sil
        self.random_state = random_state
        if flag_pca:
            ica_whiten = False
        else:
            ica_whiten = True
        self._FastICA = FastICA(n_components=n_motor_unit,
                                random_state=self.random_state,
                                max_iter=max_iter,
                                tol=tol,
                                # whiten=ica_whiten,
                                fun='cube',
                                algorithm='deflation'
                                )
        self.cashe = cashe
        if self.cashe is not None:
            if not os.path.exists('__cashe__'):
                os.mkdir('__cashe__')
            self.cashe = '__cashe__/' + self.cashe
            if not os.path.exists(self.cashe):
                os.mkdir(self.cashe)
        
        self.flag_sil = flag_sil
        self.flag_pca = flag_pca
        if self.flag_pca:
            self._PCA = PCA(n_components=n_motor_unit,
                            random_state=self.random_state,
                            whiten=True)
        
    def fit(self, emg_raw, cashe_name='all', _transform=False):
        emg_preprocessed = self._preprocess(emg_raw)
        return self._decomposition(emg_preprocessed, cashe_name, _fit=True, _transform=_transform)
        
    def transform(self, emg_raw):
        emg_preprocessed = self._preprocess(emg_raw)
        return self._decomposition(emg_preprocessed, cashe_name=None, _fit=False, _transform=True)
    
    def _preprocess(self, emg_raw):
        # extend
        emg_extended = self._extend_emg(emg_raw)
        # preprocess
        emg_centered = self._centering(emg_extended)
        return emg_centered
    
    def _decomposition(self, emg_preprocessed, cashe_name, _fit=True, _transform=True):
        # pca
        if self.flag_pca:
            if _fit:
                self._cashe_pca(emg_preprocessed, cashe_name)
            emg_preprocessed = self._PCA.transform(emg_preprocessed)
        # fastica
        if _fit:
            self._cashe_fastica(emg_preprocessed, cashe_name)
        emg_mu = self._FastICA.transform(emg_preprocessed)
        # peak detection
        if self.flag_sil or _transform:
            emg_mu_squared = np.square(emg_mu)
            spike_trains = self._EmgMu2spikeTrain(emg_mu_squared)
        # calculate SIL
        if self.flag_sil and _fit:
            self.valid_index_mu_, self.list_sil_ = self._cashe_sil(emg_mu_squared, spike_trains, cashe_name)
        # valid data
        if _transform:
            if not(self.flag_sil):
                return spike_trains, emg_mu
            else:
                st_valid = spike_trains[:, self.valid_index_mu_]
                emg_mu_valid = emg_mu[:, self.valid_index_mu_]
                return st_valid, emg_mu_valid
    
    def fit_transform(self, emg_raw, cashe_name='all'):
        return self.fit(emg_raw, cashe_name=cashe_name, _transform=True)
    
    def _extend_emg(self, emg_raw):
        df_emg_raw = pd.DataFrame(emg_raw)
        return pd.concat([df_emg_raw] + [df_emg_raw.shift(-x) for x in range(self.n_delayed)], axis=1).dropna()
    
    
    def _cashe_pca(self, emg, cashe_name):
        if self.cashe is not None:
            filepath = self.cashe + '/' + str(cashe_name) + '_pca.pickle' 
            if not(os.path.exists(filepath)):
                self._PCA.fit(emg)
                with open(filepath, 'wb') as f:
                    pickle.dump(self._PCA, f)
            else:
                with open(filepath, 'rb') as f:
                    self._PCA = pickle.load(f)
        else:
            self._PCA.fit(emg)
    
    def _cashe_fastica(self, emg, cashe_name):
        if self.cashe is not None:
            filepath = self.cashe + '/' + str(cashe_name) + '_fastica.pickle' 
            if not(os.path.exists(filepath)):
                self._FastICA.fit(emg)
                with open(filepath, 'wb') as f:
                    pickle.dump(self._FastICA, f)
            else:
                with open(filepath, 'rb') as f:
                    self._FastICA = pickle.load(f)
        else:
            self._FastICA.fit(emg)
    
    def _cashe_sil(self, emg_mu_squared, spike_trains, cashe_name):
        if self.cashe is not None:
            filepath = self.cashe + '/' + str(cashe_name) + '_sil.json' 
            if not(os.path.exists(filepath)):
                valid_index_mu, list_sil = self._sil(emg_mu_squared, spike_trains, self.threshold_sil)
                d= {'valid_index_mu': valid_index_mu, 'list_sil': list_sil}
                with open(filepath, 'w') as f:
                    json.dump(d, f, indent=4)
            else:
                with open(filepath, 'rb') as f:
                    d = json.load(f)
                valid_index_mu, list_sil = d['valid_index_mu'], d['list_sil']
        else:
            valid_index_mu, list_sil = self._sil(emg_mu_squared, spike_trains, self.threshold_sil)
            
        return valid_index_mu, list_sil
            
    def _EmgMu2spikeTrain(self, emg_squared):
        spike_trains = np.zeros_like(emg_squared)
        for i in range(emg_squared.shape[1]):
            _kmeans = KMeans(n_clusters=2, max_iter=10000, random_state=self.random_state)
            _kmeans.fit(emg_squared[:, [i]])
            
            # sort by cluster centers
            idx = np.argsort(_kmeans.cluster_centers_.sum(axis=1))
            flag = np.zeros_like(idx)
            flag[idx] = np.arange(len(idx))
            
            spike_trains[:, i] = flag[_kmeans.labels_]
        spike_trains_processsed = self._spike_post_process(emg_squared, spike_trains)
        return spike_trains_processsed

    def _spike_post_process(self, emg_squared, spike_trains):
        assert emg_squared.shape == spike_trains.shape
        # only peaks
        pre_diff = pd.DataFrame(emg_squared).diff(-1) > 0
        post_diff = pd.DataFrame(emg_squared).diff(1) > 0
        
        return spike_trains * pre_diff.values * post_diff.values
    
    def _sil(self, emg_squared, spike_trains, thre_sil):
        list_sil = []
        for i in range(emg_squared.shape[1]):
            # ignore mu that has no spike trains
            print(i)
            if np.unique(spike_trains[:, i]).shape[0] != 2:
                list_sil.append(0)
            else:
                _sil = silhouette_score(emg_squared[:, [i]], spike_trains[:, i], random_state=self.random_state)
                list_sil.append(_sil)
                
        return np.where(np.array(list_sil) >= thre_sil)[0].tolist(), list_sil
    
    @staticmethod
    def _centering(x):
        return x - np.mean(x, axis=0)
