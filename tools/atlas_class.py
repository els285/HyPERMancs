import numpy as np
import h5py
import awkward as ak
import vector
import yaml
import sys
import warnings
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)

btag_from_score = lambda array,WP : 1*(array > WP)


class HyPERParse_ATLAS:
    
    r"""
    Driving class for HyPER parser for ATLAS use
    Designed for TopCPToolkit+FastFrames output
    Requirements:
    - Selection should be made before this running (i.e. via FastFrames)
    - Neutrinos computed beforehand (point to that file with matched events)
    """
    
    # # Padding defaults
    # pad_to_jet    = 20
    # pad_to_lepton = 3
    
    def __init__(self,tree,cfg:dict):
        self.tree         = tree
        self.cfg = cfg
        self.jet_branches      = cfg["NodeFeatures"]["Jets"] 
        self.muon_branches     = cfg["NodeFeatures"]["Muons"]
        self.electron_branches = cfg["NodeFeatures"]["Electrons"]    
        self.label_branches    = cfg["Labels"]    
        
        if "Nevents" in cfg["General"].keys() and isinstance(cfg["General"]["Nevents"],int):
            self.Nevents = cfg["General"]["Nevents"]
        else:
            self.Nevents  = len(tree[self.jet_branches["pt"]].array())
            
        self.pad_to_jet    = cfg["General"].get("jet_pad" , 20)
        self.pad_to_lepton = cfg["General"].get("lepton_pad" , 3)
        
    @staticmethod
    def pad_variable(variable, max_len, pad_to = 0):
        padded_variable = ak.pad_none(variable, max_len, axis=1, clip=True)
        return ak.fill_none(padded_variable, pad_to)
        
    def read_specific_branches(self):
        
        r"""
        Loads TTree and builds arrays
        """
        self.jet_array       = self.tree.arrays(self.jet_branches.keys(),    aliases=self.jet_branches)[:self.Nevents]
        self.electron_array  = self.tree.arrays(self.electron_branches.keys(),aliases=self.electron_branches)[:self.Nevents]
        self.muon_array      = self.tree.arrays(self.muon_branches.keys(),   aliases=self.muon_branches)[:self.Nevents]

        self.input_index_array =  self.tree.arrays(self.label_branches.keys(),    aliases=self.label_branches)[:self.Nevents]

        self.lepton_array = ak.concatenate([self.electron_array,self.muon_array],axis=1)[:self.Nevents]
        self.Njets    = ak.count(self.jet_array["pt"],axis=1).to_numpy()
        self.Nleptons = ak.count(self.lepton_array["pt"],axis=1).to_numpy()
        
        
        self.jet_array["btag"] = btag_from_score(self.jet_array["btag"],-0.3780)
        
        # self.cfg["Neutrinos"]
    
    def read_generic_branches(self,tree,jet_branches,electron_branches,muon_branches, global_branches):
        
        r"""
        Reads in the TTree produced by TopCPToolkit / FastFrames.
        Builds awkward arrays of jets, leptons and neutrinos
        """
        
        self.jet_array       = tree.arrays(jet_branches.keys())
        self.electron_array  = tree.arrays(electron_branches.keys())
        self.muon_array      = tree.arrays(muon_branches.keys())
        self.global_array    = tree.arrays(global_branches.keys())
        self.lepton_array    = ak.concatenate([self.electron_array,self.muon_array],axis=1)
        

        single_neutrino = False
        # if single_neutrino:
        #     neutrino_branches = {"met":"MissingET.MET" , "phi":"MissingET.Phi"}
        #     neutrino_array = tree.arrays(neutrino_branches.keys(),aliases=neutrino_branches,cut=met_cuts)
        # else:
        #     # Import neutrino solutions here somehow
        #     pass
        #     # neutrino_solutions
        
        
        
    def target_indices(self):
        
        r"""
        Builds array for each event corresponding 
        """
        # input_index_array = self.tree.arrays(cfg["Labels"][])
        
        # Generate mask for Njets
        mask = np.arange(self.pad_to_jet) <= self.Njets[:, None]
        # Use the mask to select elements from 
        target_jet_index_array = 0*mask + -9*~mask
        target_lepton_index_array   = np.zeros([self.Nevents , self.pad_to_lepton])
        target_neutrino_index_array = np.zeros([self.Nevents , self.pad_to_lepton])
                
        hadroinc_b_mask    = self.input_index_array["hadronic_b_index"].to_numpy()
        hadronic_up_mask   = self.input_index_array["hadronic_up_index"].to_numpy()
        hadronic_down_mask = self.input_index_array["hadronic_down_index"].to_numpy()
        leptonic_b_mask    = self.input_index_array["leptonic_b_index"].to_numpy()
        
        # Positive lepton case
        positive_mask = self.lepton_array["charge"][:,0] > 0
        positive_indices = np.arange(self.Nevents)[positive_mask]
        
        # Negative lepton case
        negative_mask = self.lepton_array["charge"][:,0] < 0
        negative_indices = np.arange(self.Nevents)[negative_mask]
    
        target_jet_index_array[positive_indices,hadroinc_b_mask[positive_indices]]      = 4
        target_jet_index_array[positive_indices,hadronic_up_mask[positive_indices]]     = 5
        target_jet_index_array[positive_indices,hadronic_down_mask[positive_indices]]   = 6
        target_jet_index_array[positive_indices,leptonic_b_mask[positive_indices]]      = 1
        
        target_lepton_index_array[positive_indices,0]   = 2
        target_neutrino_index_array[positive_indices,0] = 3
        
        target_jet_index_array[negative_indices,hadroinc_b_mask[negative_indices]]      = 1
        target_jet_index_array[negative_indices,hadronic_up_mask[negative_indices]]     = 2
        target_jet_index_array[negative_indices,hadronic_down_mask[negative_indices]]   = 3
        target_jet_index_array[negative_indices,leptonic_b_mask[negative_indices]]      = 4
        
        target_lepton_index_array[negative_indices,0]   = 6
        target_neutrino_index_array[negative_indices,0] = 5
            
        self.target_jet_index_array      = target_jet_index_array
        self.target_lepton_index_array   = target_lepton_index_array
        self.target_neutrino_index_array = target_neutrino_index_array
        
        # Check whether all indices are in event. Fully_matched line found to be
        # faster than np.unique
        total_indices = np.concatenate([self.target_jet_index_array,
                                        target_lepton_index_array,
                                        target_neutrino_index_array],axis=1)
        self.fully_matched = 1*np.all(np.sort(total_indices, axis=1)[:,-6:] == np.array([1,2,3,4,5,6]), axis=1)    
                            
    # def prepare_node_data(self):
        
    #     r"""
    #     Writing data from vectors/arrays to padded numpy arrays
    #     """
        
    #     node_branches = list(set(self.jet_branches + self.muon_branches + self.electron_branches))       
    #     node_dt  = np.dtype([(br,np.float32) for br in node_branches])
        
    #     # Jets
    #     jet_data = np.zeros((len(self.njet), self.pad_to_jet), dtype=node_dt)
    #     for branch in self.jet_branches:
    #         jet_data[branch] = self.pad_variable(self.jet_array[branch], self.pad_to_jet)
    #     for branch in list(set(node_branches) - set(self.jet_branches)):
    #         jet_data[branch] = np.zeros(len(self.njet)).reshape(-1,1)
            
    #     # Leptons
    #     lepton_data = np.zeros((len(self.Nleptons), self.pad_to_lepton), dtype=node_dt)
    #     for branch in list():
    #         lepton_data[branch] = self.pad_variable(self.lepton_array[branch], self.pad_to_lepton)
    #     for branch in list(set(node_branches) - set(self.electron_branches + self.muon_branches)):
    #         lepton_data[branch] = np.zeros(len(self.Nleptons)).reshape(-1,1)
        
    #     return jet_data , lepton_data 
    
    
    def prepare_node_outputs(self):
    
        r"""
        Writing data from vectors/arrays to padded numpy arrays
        """
        
        node_dt  = np.dtype([('e', np.float32), 
                             ('eta', np.float32), 
                             ('phi', np.float32), 
                             ('pt', np.float32), 
                             ('btag', np.int32), 
                             ('charge', np.float32),
                             ('padded', np.float32)
                             ])

        # Jets
        jet_data = np.zeros((self.Nevents, self.pad_to_jet), dtype=node_dt)
        
        jet_data['pt']     = self.pad_variable(self.jet_array["pt"]  , self.pad_to_jet)
        jet_data['eta']    = self.pad_variable(self.jet_array["eta"] , self.pad_to_jet)
        jet_data['phi']    = self.pad_variable(self.jet_array["phi"] , self.pad_to_jet)
        jet_data['e']      = self.pad_variable(self.jet_array["e"]   , self.pad_to_jet)
        jet_data['btag']   = self.pad_variable(self.jet_array["btag"], self.pad_to_jet)
        jet_data['charge'] = np.zeros(self.Nevents).reshape(-1,1)
        mask = np.arange(self.pad_to_jet) <= self.Njets[:, None]
        jet_data['padded'] = np.where(mask, 1.0, np.nan)
       
        # Leptons
        lepton_data = np.zeros((self.Nevents, self.pad_to_lepton), dtype=node_dt)
        
        print(self.pad_to_lepton)

        lepton_data['pt']     = self.pad_variable(self.lepton_array["pt"]  , self.pad_to_lepton)
        lepton_data['eta']    = self.pad_variable(self.lepton_array["pt"]  , self.pad_to_lepton)
        lepton_data['phi']    = self.pad_variable(self.lepton_array["pt"]  , self.pad_to_lepton)
        lepton_data['e']      = self.pad_variable(self.lepton_array["pt"]  , self.pad_to_lepton)
        lepton_data['btag']   = np.zeros(len(self.Nleptons)).reshape(-1,1)
        lepton_data['charge'] = self.pad_variable(self.lepton_array["charge"], self.pad_to_lepton)
        mask = np.arange(self.pad_to_lepton) <= self.Nleptons[:, None]
        lepton_data['padded'] = np.where(mask, 1.0, np.nan)
        
        # For this part needs the IDs
        # lepton_data['id']     = lepton_id

        # Neutrinos        
        # neutrino_data = np.zeros((len(self.nneutrino), self.pad_to_lepton), dtype=node_dt)

        # neutrino_data['pt']     = self.neutrino_vectors_padded.pt
        # neutrino_data['eta']    = self.neutrino_vectors_padded.eta
        # neutrino_data['phi']    = self.neutrino_vectors_padded.phi
        # neutrino_data['e']      = self.neutrino_vectors_padded.pt
        # neutrino_data['btag']   = np.zeros((len(self.nneutrino), self.pad_to_neutrino), dtype=node_dt)
        # neutrino_data['charge'] = self.pad_variable(self.neutrino_array["charge"], self.pad_to_jet)
        
        # For this part needs the IDs
        # neutrino_data['id']     = neutrino_id
        
        self.jet_data    = jet_data 
        self.lepton_data = lepton_data
        
    def prepare_global_data(self):
        
        global_dt   = np.dtype([('njet', np.float32), ('nbTagged', np.float32) , ('Nleptons', np.float32)])
        global_data = np.zeros((self.Nevents, 1), dtype=global_dt)
        
        global_data["njet"] = self.Njets.reshape(-1,1)
        global_data["nbTagged"] = ak.sum(self.jet_array["btag"], axis=1).to_numpy().reshape(-1,1)
        global_data["Nleptons"] = self.Nleptons.reshape(-1,1)
        
        self.global_data = global_data
    
    def write_h5(self,outfile):
        
        r"""
        Write pre-created data to output file
        Args: outfile = name of output file
        """
        
        print('Saving data')
        with h5py.File(outfile, 'w') as h5_file:
            
            input_group = h5_file.create_group('INPUTS')
            
            input_group.create_dataset("Jets",      data=self.jet_data)
            input_group.create_dataset("Leptons",   data=self.lepton_data)
            # input_group.create_dataset("Neutrinos", data=self.neutrino_data)            
            input_group.create_dataset("Global"   , data=self.global_data)        

            global_group = h5_file.create_group('LABELS')
            
            global_group.create_dataset("Jets",        data=np.array(self.target_jet_index_array, dtype=np.int64))
            global_group.create_dataset("Leptons",     data=np.array(self.target_lepton_index_array, dtype=np.int64))
            # global_group.create_dataset("NeutrinoNodeID",   data=np.array(self.target_neutrino_index_array, dtype=np.int64))

            global_group.create_dataset("FullyMatched", data = self.fully_matched) # To be depreciated

        print('Programme finished')