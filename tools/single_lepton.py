import vector
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import curve_fit



def vectorised(lepton_vector,bjet_vectors,met_array):
    
    # How to do this in a vectorised way?
    # Apply a mask based on discriminant 
    mW = 80.377
    A = lepton_vector.pt**2
    B = 2*lepton_vector.pz*(mW**2/2 + lepton_vector.pt*met_array["pt"])
    C = mW**4/4 + mW**2*lepton_vector.pt*met_array["pt"] - (lepton_vector.pt**2)*(met_array["pt"]**2)
    
    discrim = B**2 - 4*A*C 
    
    solvable = discrim >= 0
    
    pZnu_plus  = (-B[solvable] + discrim[solvable]**0.5)/(2*A[solvable])
    pZnu_minus = (-B[solvable] - discrim[solvable]**0.5)/(2*A[solvable])
    
    
    
def generate_2D_Gaussian(met_array):
    
    """
    Straight from ChatGPT
    """
    px = met_array["pt"].to_numpy()*np.cos(met_array["phi"].to_numpy())
    py = met_array["pt"].to_numpy()*np.sin(met_array["phi"].to_numpy())

    # Sample data: replace these with your actual data
    arrayX = px  # Example data for X
    arrayY = py  # Example data for Y

    # Compute the covariance matrix
    data = np.vstack((arrayX, arrayY))
    cov_matrix = np.cov(data)

    # Extract variances and covariance
    sigma_x = np.sqrt(cov_matrix[0, 0])  # Standard deviation for X
    sigma_y = np.sqrt(cov_matrix[1, 1])  # Standard deviation for Y
    cov_xy = cov_matrix[0, 1]             # Covariance between X and Y

    # Mean of the data
    mean = [np.mean(arrayX), np.mean(arrayY)]

    # Construct the covariance matrix for the 2D Gaussian
    cov = [[sigma_x**2, cov_xy], 
        [cov_xy, sigma_y**2]]  # Covariance matrix

    # Create a grid for the 2D Gaussian reconstruction
    x_range = np.linspace(mean[0] - 4 * sigma_x, mean[0] + 4 * sigma_x, 100)
    y_range = np.linspace(mean[1] - 4 * sigma_y, mean[1] + 4 * sigma_y, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Generate the 2D Gaussian values on the grid
    Z = multivariate_normal(mean, cov).pdf(np.dstack((X, Y)))


def loop(lepton_vector,bjet_vector,met_array,mean,cov):
    
    for i in range(len(lepton_vector)):
        
        e_lepton = lepton_vector[i]
        e_bjet   = bjet_vector[i]
        e_met    = met_array[i]
                
        mW = 80.377
        A = e_lepton.pt**2
        B = 2*e_lepton.pz*(mW**2/2 + e_lepton.pt*e_met["pt"])
        C = mW**4/4 + mW**2*e_lepton.pt*e_met["pt"] - (e_lepton.pt**2)*(e_met["pt"]**2)

        discrim = B**2 - 4*A*C 
        
        if discrim>=0:
            
            pZnu_plus  = (-B + discrim**0.5)/(2*A)
            pZnu_minus = (-B - discrim**0.5)/(2*A)
            
            plus_neutrino  = vector.zip({"pt":e_met["pt"] , "pz":pZnu_plus,  "m":0})
            minus_neutrino = vector.zip({"pt":e_met["pt"] , "pz":pZnu_minus, "m":0})
            
            top_plus  = (plus_neutrino + e_lepton + e_bjet)
            top_minus = (minus_neutrino + e_lepton + e_bjet)
            
            if abs(172.5 - top_plus.m) < abs(172.5 - top_minus.m):
                pZnu =  pZnu_plus
            else:
                pZnu = pZnu_minus
                
        if discrim<0:
            
            # Smear the transverse kinematics slightly
            pass
            

            
    
        