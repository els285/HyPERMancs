import yaml
import sys 
import uproot       

from atlas_class import HyPERParse_ATLAS
        
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
        
        