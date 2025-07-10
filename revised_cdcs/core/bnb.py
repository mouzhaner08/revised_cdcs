import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import eval_hermite
from typing import List, Tuple, Dict, Optional
from core.bnb_helper_anm import bnb_helper_anm
from core.testAn import compute_test_tensor_G

def map_var_to_indices(variables, K):
    """
    Placeholder for R's mapVarToInd
    Get Hermite basis column indices
    """
    return [i * K + d for i in variables for d in range(K)]


def conv_to_count(pvals, bs):
    """
    Placeholder for R's convToCont
    Convert discrete p-values to cts uniforms
    """
    cut_points = np.linspace(0, 1, bs+2)
    indices = (pvals * bs).astype(int)
    return np.random.uniform(cut_points[indices], cut_points[indices + 1])


class ConfidenceSet:
    """
    Branch and bound procedure for producing a confidence set of causal orderings
    Assumes an additive noise model where the basis is either bsplines or polynomial
    
    Parameters:
    -----------
    Y : An n by p matrix with the observations
    G : A 3d array containing the evaluated test functions. G[i, j, u] is h_j(Y_{u,i})
    bs : int, default=200
        The number of bootstrap resamples for the null distribution
    agg_type : int or str, default=3
        The aggregation used for the test statistic:
            - 1: sum_{u ≺ pr(v), j} |h_j(Y_u)^T η_v|
            - 2: sum_{u ≺ pr(v), j} |h_j(Y_u)^T η_v|^2
            - 3 or "inf": max_{u ≺ pr(v), j} |h_j(Y_u)^T η_v|
    alpha : float, default=0.05
        The level of the confidence set
    basis : str, default="bspline"
        The basis used to model f_v. Choices are "bspline" or "poly"
    K : int, default=5
        Either the df for bsplines or the number of polynomial terms
        
    Returns:
    --------
    pd.DataFrame
        A dataframe with p-values and corresponding orderings
    """
    def __init__(self, Y:np.ndarray, bs: int = 200,
                 alpha: float = 0.05, basis: str = 'bspline',agg_type: int = 3, p_value_agg: str = "tippett", 
                 K: int = 5, intercept: bool = True, verbose: bool = True):
        self.Y = Y
        self.G = compute_test_tensor_G(self.Y) # shape (n, k=7, p)
        self.bs = bs
        self.alpha = alpha
        self.basis = basis
        self.K = K # total basis terms across all vars
        self.k = self.G.shape[1] # num of test functions
        self.agg_type = 3 if agg_type == 'inf' else agg_type
        self.p_value_agg = p_value_agg
        self.intercept = intercept
        self.verbose = verbose
        self.n, self.p = self.Y.shape
        self.ancest_mat = self._build_ancestral_matrix()
        self.results: Optional[pd.DataFrame] = None
        self.cutoff, self.initial_track = self._calculate_cutoff()
        self.bootstrap_indices = [np.random.choice(self.n, self.n, replace=True) for _ in range(self.bs)] # shape (bs, n)

    def _calculate_cutoff(self) -> Tuple[float, np.ndarray]:
        """
        Calculate cutoff and initial tracking values
        """
        if self.p_value_agg == "fisher":
            cutoff = stats.chi2.ppf(1 - self.alpha, df=2 * (self.p - 1))
            current_track = np.zeros(self.p)
        else:
            cutoff = stats.beta.ppf(self.alpha, 1, self.p - 1)
            current_track = np.ones(self.p)
        
        return cutoff, current_track
    
    def _get_hermite(self, x: np.ndarray, K: int) -> np.ndarray:
        """
        Generate Hermite polynomial basis functions up to degree K - 1.
        Equivalent to cdcs::getHermite in R.
        """
        return np.column_stack([eval_hermite(d, x) for d in range(K)])
    
    def _build_ancestral_matrix(self) -> np.ndarray:
        """
        Build the ancestral matrix based on the specified basis
        """
        ancest_mat = np.zeros((self.n, self.p * self.K))
        if self.basis == "poly":
            for i in range(self.p):
                ancest_mat[:, (self.K * i):(self.K * (i + 1))] = self._get_hermite(self.Y[:, i], self.K)
        
        elif self.basis == "bspline":
            for i in range(self.p):
                ancest_mat[:, (self.K * i):(self.K * (i + 1))] = self._bs_basis(self.Y[:, i], self.K)
        
        return ancest_mat

    def _update_p_vals(self, native_ind: int, hash_ind: List[int], current_seq: pd.DataFrame,
                       unique_res: List[Dict]) -> pd.DataFrame:
        """
        Helper function to update the counter for p-values
        """
        ### Get the set and trackers from the larger function environment 
        match_ind = hash_ind[native_ind]
        current_seq_i = list(current_seq.iloc[native_ind, 1:])
        current_track = current_seq.iloc[native_ind, 0]
        
        ## get P-values computed from function environment ##
        fresh_p_vals = list(unique_res[match_ind]["pVals"])
        possible_children = list(unique_res[match_ind]["possibleChildren"])
        
        # update orderings using either fisher or tippett
        new_track = np.minimum(current_track, fresh_p_vals)
        continue_on = new_track > self.cutoff
        
        # Check if there are any orderings which haven't passed the cut-off 
        # i.e., still viable orderings
        if np.sum(continue_on) > 0:
            
            # data frame to return
            # Col 1: updated tracker
            # Remaining columns: Orderings which have not passed the cut-off
            valid_indices = np.where(continue_on)[0]
            new_track_valid = new_track[valid_indices]
            possible_children_valid = np.array(possible_children)[valid_indices]
            
            # Create matrix with repeated current_seq_i
            n_valid = len(valid_indices)
            seq_matrix = np.tile(current_seq_i, (n_valid, 1))
            
            # Combine new_track, seq_matrix, and possible_children
            new_data = np.column_stack([
                new_track_valid,
                seq_matrix,
                possible_children_valid
            ])
            
            new_seq = pd.DataFrame(new_data)
            
        else:
            # if no orderings have passed the cut-off, return an empty data frame    
            new_seq = pd.DataFrame(np.zeros((0, len(current_seq_i) + 2)))
        
        # names for each column
        col_names = ["currentTrack"] + [f"V{i+1}" for i in range(len(current_seq_i) + 1)]
        if not new_seq.empty:
            new_seq.columns = col_names
        
        new_seq.reset_index(drop=True, inplace=True)
        
        return new_seq
    
    def _test_ancest(self, ancest: List[int]) -> Dict[str, List]:
        """
        Takes in a set of ancestors and tests whether any node not included in the set 
        could be a descendant of the ancestors
        """
        
        ## will come in as a list because gets pulled out of data.frame
        ancest = list(ancest)
        
        ## set of any nodes not in the set which could be potential children
        possible_children = [i for i in range(1, self.p+1) if i not in ancest]
        
        ## subtract 1 for cpp indexing
        ancest_idx = [a - 1 for a in ancest]  # convert to 0-based
        mapped_indices = map_var_to_indices(ancest_idx, self.K)
        
        # Get the ancestral matrix columns and Y columns for possible children
        ancest_mat_subset = self.ancest_mat[:, mapped_indices] if mapped_indices else self.ancest_mat[:, :0]
        Y_children = self.Y[:, [i-1 for i in possible_children]] if possible_children else self.Y[:, :0]

        G_list = [np.repeat(self.G[:, :, i][:, :, np.newaxis], self.K, axis=2) for i in ancest_idx]
        G_ancest = np.concatenate(G_list, axis=2) if G_list else self.G[:, :, :0]
        
        p_vals_result = bnb_helper_anm(ancest=ancest_mat_subset, children=Y_children, G=G_ancest,
                                           withinAgg=self.agg_type, aggType=self.agg_type, 
                                           bs=self.bs, intercept=self.intercept,
                                           bootstrap_indices=self.bootstrap_indices)
        ## data frame with:
        ## Column 1: p-values (converted to uniform)
        ## Column 2: possible children
        ret = {
            "pVals": conv_to_count(p_vals_result, self.bs),
            "possibleChildren": possible_children
        }
        return ret

    def branchAndBound(self) -> pd.DataFrame:
        """
        Run the branch and bound algorithm to find confidence sets.
        
        Returns:
        --------
        pd.DataFrame
            A dataframe with p-values and corresponding orderings
        """
        
        # Initial values
        # Start with each node and initial currentTrack value
        initial_data = []
        for i in range(self.p):
            initial_data.append([self.initial_track[i], i + 1])
        
        current_seq = pd.DataFrame(initial_data, columns=["currentTrack", "V1"])
        
        while len(current_seq) > 0 and current_seq.shape[1] <= self.p:
            
            # hash involves all sequences which are the same set
            # computation only depends on v and set an(v), so the "ordering" of an(v) doesn't matter
            hash_values = []
            for idx in range(len(current_seq)):
                seq_vals = current_seq.iloc[idx, 1:].values
                hash_val = ".".join(map(str, sorted(seq_vals)))
                hash_values.append(hash_val)
            
            # uniqueHash is the list of unique ancestral sets
            unique_hash = list(set(hash_values))
            
            # each ongoing ordering maps to a specific set
            hash_ind = [unique_hash.index(h) for h in hash_values]
            
            if self.verbose:
                print("==========================================================")
                print(f"Current Order: {current_seq.shape[1] - 1}")
                perm_count = len(current_seq)
                # Calculate the proportion (equivalent to R's prod calculation)
                order_size = current_seq.shape[1] - 2
                if order_size >= 0:
                    total_perms = 1
                    for i in range(order_size + 1):
                        total_perms *= (self.p - i)
                    proportion = round(perm_count / total_perms, 3) if total_perms > 0 else 0
                else:
                    proportion = 0
                print(f"Number of Perm: {perm_count} ( {proportion} )")
                print(f"Number of Comb: {len(unique_hash)}")
                print()
            
            # For each value in uniqueHash, get a representative set and run .testAncest on that set
            unique_res = []
            for unique_h in unique_hash:
                # Find first occurrence of this hash
                representative_idx = hash_values.index(unique_h)
                ancest = current_seq.iloc[representative_idx, 1:].values.astype(int).tolist()
                result = self._test_ancest(ancest)
                unique_res.append(result)
            
            ### Update Sequences ###
            updated_seq = []
            for i in range(len(hash_values)):
                updated = self._update_p_vals(i, hash_ind, current_seq, unique_res)
                updated_seq.append(updated)
            
            # Combine all updated sequences (equivalent to data.table::rbindlist)
            if any(not df.empty for df in updated_seq):
                current_seq = pd.concat([df for df in updated_seq if not df.empty], ignore_index=True)
            else:
                current_seq = pd.DataFrame()
        
        ## Check if there are any orderings to return
        if len(current_seq) == 0:
            # return empty data frame
            current_seq = pd.DataFrame(np.zeros((0, self.p + 1)))
        else:
            ## if there are orderings to be returned
            ## Col 1: final p-value of aggregated p-values 
            ## then form data frame with orderings
            if self.p_value_agg == "fisher":
                p_vals = stats.chi2.sf(current_seq.iloc[:, 0], df=2 * (self.p - 1))
            else:
                p_vals = stats.beta.cdf(current_seq.iloc[:, 0], 1, self.p - 1)
            
            current_seq.iloc[:, 0] = p_vals
        
        # Set column names
        col_names = ["pValue"] + [f"V{i+1}" for i in range(self.p)]
        if not current_seq.empty:
            current_seq.columns = col_names[:current_seq.shape[1]]
        else:
            current_seq = pd.DataFrame(columns=col_names)
        
        # Store results
        self.results = current_seq
        return current_seq