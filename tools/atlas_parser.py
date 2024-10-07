import uproot 
import numpy as np
from tqdm import tqdm
import h5py
import awkward as ak
import vector
import yaml
import sys
import warnings
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)


class HyPERParse_ATLAS:
    
    r"""
    Driving class for HyPER parser for ATLAS use
    Designed for TopCPToolkit+FastFrames output
    Requirements:
    - Selection should be made before this running (i.e. via FastFrames)
    - Neutrinos computed beforehand (point to that file with matched events)
    """
    
    pad_to_jet    = 20
    pad_to_lepton = 3
    
    def __init__(self,tree,cfg):
        self.tree         = tree
        self.cfg = cfg
        self.jet_branches      = cfg["NodeFeatures"]["Jets"] 
        self.muon_branches     = cfg["NodeFeatures"]["Muons"]
        self.electron_branches = cfg["NodeFeatures"]["Electrons"]        
        
        self.Nevents  = len(tree[self.jet_branches["pt"]].array())
        self.Njets    = ak.count(tree[self.jet_branches["pt"]].array(),axis=1).to_numpy()
        self.Nleptons = None
        

    @staticmethod
    def pad_variable(variable, max_len, pad_to = 0):
        padded_variable = ak.pad_none(variable, max_len, axis=1, clip=True)
        return ak.fill_none(padded_variable, pad_to)
        
    # @staticmethod  
    # def compute_single_lepton_eta(lepton_vector,met_array):
    #     # How to do this in a vectorised way?
    #     # Apply a mask based on discriminant 
    #     mW = 80.377
    #     A = lepton_vector.pt**2
    #     B = 2*lepton_vector.pz*(mW**2/2 + lepton_vector.pt*met_array["pt"])
    #     C = mW**4/4 + mW**2*lepton_vector.pt*met_array["pt"] - (lepton_vector.pt**2)*(met_array["pt"]**2)
        
    #     discrim_mask = B**2 - 4*A*C >= 0 
        
    #     pZnu_plus  = (-B + (B**2 - 4*A*C)**(0.5))/(2*A)
    #     pZnu_minus = (-B - (B**2 - 4*A*C)**(0.5))/(2*A)
    
    
    def read_specific_branches(self):
        

        self.jet_array       = self.tree.arrays(self.jet_branches.keys(),    aliases=self.jet_branches)
        self.electron_array  = self.tree.arrays(self.electron_branches.keys(),aliases=self.electron_branches)
        self.muon_array      = self.tree.arrays(self.muon_branches.keys(),   aliases=self.muon_branches)
        
        self.lepton_array = ak.concatenate([self.electron_array,self.muon_array],axis=1)
        
        self.Nleptons = ak.count(self.lepton_array["pt"],axis=1).to_numpy()
        
    
        # self.cfg["Neutrinos"]
    
        
    def read_generic_branches(self,tree,jet_branches,electron_branches,muon_branches, global_branches):
        """
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
        input_index_array = self.tree.arrays(["parton_truth_had_b_index",
                                           "parton_truth_down_index" ,  
                                           "parton_truth_up_index",
                                           "parton_truth_lep_b_index" ])
        
        # Generate mask for Njets
        mask = np.arange(self.pad_to_jet) <= self.Njets[:, None]
        # Use the mask to select elements from 
        target_jet_index_array = 0*mask + -9*~mask
        target_lepton_index_array   = np.zeros([self.Nevents , self.pad_to_lepton])
        target_neutrino_index_array = np.zeros([self.Nevents , self.pad_to_lepton])
                
        m1 = input_index_array["parton_truth_had_b_index"].to_numpy()
        m2 = input_index_array["parton_truth_up_index"].to_numpy()
        m3 = input_index_array["parton_truth_down_index"].to_numpy()
        m4 = input_index_array["parton_truth_lep_b_index"].to_numpy()
        
        # m1[m1==-1]=0
        # m2[m2==-1]=0
        # m3[m3==-1]=0
        # m4[m4==-1]=0
        
        # Positive lepton case
        positive_mask = self.lepton_array["charge"][:,0] > 0
        positive_indices = np.arange(self.Nevents)[positive_mask]
        
        # Negative lepton case
        negative_mask = self.lepton_array["charge"][:,0] < 0
        negative_indices = np.arange(self.Nevents)[negative_mask]
    
        target_jet_index_array[positive_indices,m1[positive_indices]] = 4
        target_jet_index_array[positive_indices,m2[positive_indices]] = 5
        target_jet_index_array[positive_indices,m3[positive_indices]] = 6
        target_jet_index_array[positive_indices,m4[positive_indices]] = 1
        
        target_lepton_index_array[positive_indices,0]   = 2
        target_neutrino_index_array[positive_indices,0] = 3
        
        target_jet_index_array[negative_indices,m1[negative_indices]] = 1
        target_jet_index_array[negative_indices,m2[negative_indices]] = 2
        target_jet_index_array[negative_indices,m3[negative_indices]] = 3
        target_jet_index_array[negative_indices,m4[negative_indices]] = 4
        
        target_lepton_index_array[negative_indices,0]   = 6
        target_neutrino_index_array[negative_indices,0] = 5
            
        self.target_jet_index_array      = target_jet_index_array
        self.target_lepton_index_array   = target_lepton_index_array
        self.target_neutrino_index_array = target_neutrino_index_array
                        
    
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
        
        node_dt  = np.dtype([('e', np.float32), ('eta', np.float32), ('phi', np.float32), ('pt', np.float32), ('btag', np.int32), ('charge', np.float32)])

        # Jets
        jet_data = np.zeros((len(self.Njets), self.pad_to_jet), dtype=node_dt)
        
        jet_data['pt']     = self.pad_variable(self.jet_array["pt"]  , self.pad_to_jet)
        jet_data['eta']    = self.pad_variable(self.jet_array["eta"] , self.pad_to_jet)
        jet_data['phi']    = self.pad_variable(self.jet_array["phi"] , self.pad_to_jet)
        jet_data['e']      = self.pad_variable(self.jet_array["e"]   , self.pad_to_jet)
        jet_data['btag']   = self.pad_variable(self.jet_array["btag"], self.pad_to_jet)
        jet_data['charge'] = np.zeros(len(self.Njets)).reshape(-1,1)
        
        # Leptons
        lepton_data = np.zeros((len(self.Nleptons), self.pad_to_lepton), dtype=node_dt)

        lepton_data['pt']     = self.pad_variable(self.lepton_array["pt"]  , self.pad_to_lepton)
        lepton_data['eta']    = self.pad_variable(self.lepton_array["pt"]  , self.pad_to_lepton)
        lepton_data['phi']    = self.pad_variable(self.lepton_array["pt"]  , self.pad_to_lepton)
        lepton_data['e']      = self.pad_variable(self.lepton_array["pt"]  , self.pad_to_lepton)
        lepton_data['btag']   = np.zeros(len(self.Nleptons)).reshape(-1,1)
        lepton_data['charge'] = self.pad_variable(self.lepton_array["charge"], self.pad_to_lepton)
        
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
        global_data["nbTagged"] = np.sum(self.jet_array["btag"], axis=1).to_numpy().reshape(-1,1)
        global_data["Nleptons"] = self.Nleptons.reshape(-1,1)
        
        self.global_data = global_data
    
    def write_h5(self,outfile):
        
        print('Saving data')
        with h5py.File(outfile, 'w') as h5_file:
            
            input_group = h5_file.create_group('INPUTS')
            
            input_group.create_dataset("Jets",      data=self.jet_data)
            input_group.create_dataset("Leptons",   data=self.lepton_data)
            # input_group.create_dataset("Neutrinos", data=self.neutrino_data)            
            input_group.create_dataset("Global"   , data=self.global_data)        

            global_group = h5_file.create_group('LABELS')
            
            global_group.create_dataset("JetNodeID",        data=np.array(self.target_jet_index_array, dtype=np.int64))
            global_group.create_dataset("LeptonNodeID",     data=np.array(self.target_lepton_index_array, dtype=np.int64))
            # global_group.create_dataset("NeutrinoNodeID",   data=np.array(self.target_neutrino_index_array, dtype=np.int64))

            # global_group.create_dataset("FullyMatched", data = np.array(IndexSelect, dtype= np.int32)) # To be depreciated

        print('program finished')
        
        
        
def check_required_keys(config, keys):
    missing_keys = []
    for key in keys:
        if key not in config:
            missing_keys.append(key)
    if missing_keys:
        raise KeyError(f"Missing required config keys: {', '.join(missing_keys)}")
    
        
        
def main():
    
    with open(sys.argv[1]) as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
              
    try:
        check_required_keys(cfg["NodeFeatures"]["Jets"], ["pt","eta","phi","e","btag"])
    except KeyError as e:
        print(e)
        
    try:
        check_required_keys(cfg["NodeFeatures"]["Electrons"], ["pt","eta","phi","e","charge"])
    except KeyError as e:
        print(e)
        
    try:
        check_required_keys(cfg["NodeFeatures"]["Muons"], ["pt","eta","phi","e","charge"])
    except KeyError as e:
        print(e)
        
    
    inputfile = cfg["Files"]["input"]
    tree      = cfg["Files"]["tree"]
    input_tree = uproot.open(f"{inputfile}:{tree}")
    
    Parser_Object = HyPERParse_ATLAS(tree=input_tree, cfg=cfg)
    
    Parser_Object.read_specific_branches()
    
    Parser_Object.prepare_node_outputs()
    
    Parser_Object.prepare_global_data()
    
    Parser_Object.target_indices()
            
    Parser_Object.write_h5(cfg["Files"]["output"])
    
    
if __name__ == "__main__":
    main()
        
        