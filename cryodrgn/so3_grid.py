'''
Implementation of Yershova et al. "Generating uniform incremental
grids on SO(3) using the Hopf fribration"
'''

import numpy as np
import os
import json
import torch
from . import lie_tools

def grid_s1(resol):
    Npix = 6*2**resol
    dt = 2*np.pi/Npix
    grid = np.arange(Npix)*dt + dt/2
    print('number of s1 grid is ',len(grid))
    return grid

def grid_s2(resol, theta_prior=None,theta_range=None):
    Nside = 2**resol
    Npix = 12*Nside*Nside
    theta, phi = pix2ang(Nside, np.arange(Npix), nest=True)

   #if theta_prior is not None:
   #    # try to confine the search range.
   #    print('number of theta before',len(theta))
   #    pose_np=np.array([theta,phi])
   #    #index = (pose_np[0, :] >= 4 * np.pi / 9) & (pose_np[0, :] <= 5 * np.pi / 9)
   #    #index = (pose_np[0, :] >= theta_prior-theta_range) & (pose_np[0, :] <= theta_prior + theta_range)
   #    index = pose_np[0, :] == np.pi / 2
   #    pose_np=pose_np[:,index]
   #    theta=pose_np[0]
   #    phi=pose_np[1]
   #    print('number of theta after', len(theta))

    return theta, phi

def hopf_to_quat(theta, phi, psi):
    '''
    Hopf coordinates to quaternions
    theta: [0,pi)
    phi: [0, 2pi)
    psi: [0, 2pi)
    '''
    ct = np.cos(theta/2)
    st = np.sin(theta/2)
    quat = np.array([ct*np.cos(psi/2),
                     ct*np.sin(psi/2),
                     st*np.cos(phi+psi/2),
                     st*np.sin(phi+psi/2)])
    return quat.T.astype(np.float32)

def grid_SO3(resol,theta_prior=None, theta_range=np.pi/18):
    theta, phi = grid_s2(resol, theta_prior=theta_prior, theta_range=theta_range)
    psi = grid_s1(resol)
    quat = hopf_to_quat(np.repeat(theta,len(psi)), # repeats each element by len(psi)
                        np.repeat(phi,len(psi)), # repeats each element by len(psi)
                        np.tile(psi,len(theta))) # tiles the array len(theta) times
    return quat #hmm convert to rot matrix?

def s2_grid_SO3(resol,theta_prior=None, theta_range=np.pi/18):
    theta, phi = grid_s2(resol, theta_prior=theta_prior, theta_range=theta_range)
    quat = hopf_to_quat(theta, phi, np.zeros((len(phi),)))
    return quat

#### Neighbor finding ####

def get_s1_neighbor(mini, curr_res):
    '''
    Return the 2 nearest neighbors on S1 at the next resolution level
    '''
    Npix = 6*2**(curr_res+1)
    dt = 2*np.pi/Npix
    #return np.array([2*mini, 2*mini+1])*dt + dt/2
    # the fiber bundle grid on SO3 is weird
    # the next resolution level's nearest neighbors in SO3 are not
    # necessarily the nearest neighbor grid points in S1
    # include the 13 neighbors for now... eventually learn/memoize the mapping
    ind = np.arange(2*mini-1, 2*mini+3)
    if ind[0] < 0:
        ind[0] += Npix
    return ind*dt+dt/2, ind

def get_s2_neighbor(mini, cur_res):
    '''
    Return the 4 nearest neighbors on S2 at the next resolution level
    '''
    Nside = 2**(cur_res+1)
    ind = np.arange(4)+4*mini
    return pix2ang(Nside, ind, nest=True), ind

def get_base_ind(ind, base):
    '''
    Return the corresponding S2 and S1 grid index for an index on the base SO3 grid
    '''
    Np = 6 * 2**base
    psii = ind % Np
    thetai = ind // Np
    return np.stack((thetai, psii), axis=1)


def get_neighbor(quat, s2i, s1i, cur_res):
    '''
    Return the 8 nearest neighbors on SO3 at the next resolution level
    '''
    (theta, phi), s2_nexti = get_s2_neighbor(s2i, cur_res)
    psi, s1_nexti = get_s1_neighbor(s1i, cur_res)
    quat_n = hopf_to_quat(np.repeat(theta,len(psi)),
                          np.repeat(phi,len(psi)),
                          np.tile(psi,len(theta)))
    ind = np.array([np.repeat(s2_nexti,len(psi)),
                    np.tile(s1_nexti, len(theta))])
    ind = ind.T
    # find the 8 nearest neighbors of 16 possible points
    # need to check distance from both +q and -q
    dists = np.minimum(np.sum((quat_n-quat)**2,axis=1), np.sum((quat_n+quat)**2,axis=1))
    ii = np.argsort(dists)[:8]
    return quat_n[ii], ind[ii]


try:
    with open(f"{os.path.dirname(__file__)}/healpy_grid.json") as hf:
        _GRIDS = {int(k): np.array(v).T for k, v in json.load(hf).items()}
except IOError as e:
    print("WARNING: Couldn't load cached healpy grid; will fall back to importing healpy")
    _GRIDS = None

def pix2ang(Nside, ipix, nest=False, lonlat=False):
    if _GRIDS is not None and Nside in _GRIDS and nest and not lonlat:
        return _GRIDS[Nside][ipix].T
        return ret.T
    else:
        try:
            import healpy
        except ImportError:
            raise RuntimeError("You need to `pip install healpy` to run with non-standard grid sizes.")
        return healpy.pix2ang(Nside, ipix, nest=nest, lonlat=lonlat)


def gaussian_sample(mean, sigma, interval=5):
    from scipy.stats import norm, truncnorm

    n = interval

    probabilities = np.linspace(0, 1, n + 1)

    # Compute the quantiles (interval boundaries) for the given probabilities
    quantiles = norm.ppf(probabilities, loc=mean, scale=sigma)

    # Calculate the mean position within each interval
    means = []
    for i in range(n):
        a, b = quantiles[i], quantiles[i + 1]
        # Standardize the interval limits for the truncated normal distribution
        a_std, b_std = (a - mean) / sigma, (b - mean) / sigma
        # Calculate the mean within the interval
        interval_mean = truncnorm.mean(a_std, b_std, loc=mean, scale=sigma)
        means.append(interval_mean)

    filtered_means = np.array([m for m in means if (mean - 2 * sigma) <= m <= (mean + 2 * sigma)])
    return filtered_means


def gaussian_sample_torch(mean, sigma, interval=15, device=None):
    import math
    n = interval

    # Ensure mean and sigma are tensors on the specified device
    mean = torch.tensor(mean, device=device, dtype=torch.float32)
    sigma = torch.tensor(sigma, device=device, dtype=torch.float32)

    # Reshape mean and sigma to (B, 1) for broadcasting
    mean = mean.view(-1, 1)  # Shape: (B, 1)
    sigma = sigma.view(-1, 1)  # Shape: (B, 1)

    B = mean.size(0)  # Batch size

    # Generate probabilities
    probabilities = torch.linspace(0, 1, n + 1, device=device, dtype=torch.float32)
    probabilities = probabilities.view(1, -1)  # Shape: (1, n + 1)

    # Avoid probabilities at 0 and 1 to prevent infinities in inverse CDF
    epsilon = 1e-10
    probabilities = torch.clamp(probabilities, epsilon, 1 - epsilon)

    # Compute quantiles (interval boundaries) using the inverse CDF of the normal distribution
    sqrt_2 = torch.tensor(math.sqrt(2), device=device, dtype=torch.float32)
    erfinv_arg = 2 * probabilities - 1  # Shape: (1, n + 1)
    erfinv_result = torch.erfinv(erfinv_arg)  # Shape: (1, n + 1)

    quantiles = mean + sigma * sqrt_2 * erfinv_result  # Shape: (B, n + 1)

    # Standardize the interval limits for the truncated normal distribution
    a = quantiles[:, :-1]  # Shape: (B, n)
    b = quantiles[:, 1:]  # Shape: (B, n)
    a_std = (a - mean) / sigma  # Shape: (B, n)
    b_std = (b - mean) / sigma  # Shape: (B, n)

    # Compute the cumulative distribution function (CDF) for the standard normal distribution
    def standard_normal_cdf(x):
        sqrt_2 = torch.tensor(math.sqrt(2), device=x.device, dtype=x.dtype)
        return 0.5 * (1 + torch.erf(x / sqrt_2))

    p_a = standard_normal_cdf(a_std)  # Shape: (B, n)
    p_b = standard_normal_cdf(b_std)  # Shape: (B, n)

    # Compute the probability density function (PDF) for the standard normal distribution
    def standard_normal_pdf(x):
        sqrt_two_pi = torch.tensor(math.sqrt(2 * math.pi), device=x.device, dtype=x.dtype)
        return torch.exp(-0.5 * x ** 2) / sqrt_two_pi

    psi_a = standard_normal_pdf(a_std)  # Shape: (B, n)
    psi_b = standard_normal_pdf(b_std)  # Shape: (B, n)

    # Compute the mean of the truncated normal distribution within each interval
    numerator = psi_a - psi_b  # Shape: (B, n)
    denominator = p_b - p_a  # Shape: (B, n)

    # Handle potential division by zero using torch.where
    zero_denom = denominator == 0  # Shape: (B, n)
    safe_denominator = torch.where(zero_denom, torch.ones_like(denominator), denominator)

    interval_means = mean + (numerator / safe_denominator) * sigma  # Shape: (B, n)
    midpoint = 0.5 * (a + b)  # Shape: (B, n)
    interval_means = torch.where(zero_denom, midpoint, interval_means)

    # Filter means within [mean - 2 * sigma, mean + 2 * sigma]
    lower_bound = mean - 2 * sigma  # Shape: (B, 1)
    upper_bound = mean + 2 * sigma  # Shape: (B, 1)
    mask = (interval_means >= lower_bound) & (interval_means <= upper_bound)  # Shape: (B, n)

    # Collect filtered means for each batch element
    filtered_means_list = []

    for i in range(B):
        interval_means_i = interval_means[i]  # Shape: (n,)
        mask_i = mask[i]  # Shape: (n,)
        filtered_means_i = interval_means_i[mask_i]  # Variable length
        filtered_means_list.append(filtered_means_i)

    filtered_means_list = torch.stack(filtered_means_list)

    return filtered_means_list


def helical_sample_grid_so3(sampling_steps, tilt_sigma, psi_sigma):
    # create the sample grid of rotation (N) and tilt (M)

    rot_list = torch.arange(0, 360, sampling_steps) * torch.pi / 180

    tilt_list = torch.tensor(gaussian_sample(90, tilt_sigma, 7) * torch.pi / 180, dtype=torch.float32)

    psi_list_1 = torch.tensor(gaussian_sample(0, psi_sigma) * torch.pi / 180, dtype=torch.float32)

    psi_list_2 = torch.tensor(gaussian_sample(180, psi_sigma) * torch.pi / 180, dtype=torch.float32)

    psi_list = torch.cat((psi_list_1, psi_list_2))

    # create a (MxN) meshgrid with N rotation around the z axis and M tilt angle

    rot, tilt, psi = torch.meshgrid(rot_list, tilt_list, psi_list)

    angle_list = torch.stack([rot, tilt, psi], dim=-1)

    sample_shape = angle_list.shape

    # convert the rot tilt psi to angle

    rot_all = angle_list[..., 0].view(-1)
    tilt_all = angle_list[..., 1].view(-1)
    psi_all = angle_list[..., 2].view(-1)

    B = len(rot_all)

    all_matrix = []

    for i in range(B):
        ca, sa = torch.cos(rot_all)[i], torch.sin(rot_all)[i]
        cb, sb = torch.cos(tilt_all)[i], torch.sin(tilt_all)[i]
        cy, sy = torch.cos(psi_all)[i], torch.sin(psi_all)[i]

        Ra = torch.tensor([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        Rb = torch.tensor([[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]])
        Ry = torch.tensor(([cy, -sy, 0], [sy, cy, 0], [0, 0, 1]))
        R = Ry @ Rb @ Ra
        all_matrix.append(R)

    all_matrix = torch.stack(all_matrix, dim=0)
    all_matrix = all_matrix.reshape((sample_shape[0], sample_shape[1], sample_shape[2], 3, 3))

    all_angle_shape = all_matrix.shape[:3]
    print('using rot, tilt, psi number of ',all_matrix.shape[:3])

    return all_matrix, all_angle_shape


def helical_sample_grid(sampling_steps, tilt_sigma):
    # create the sample grid of rotation (N) and tilt (M)

    rot_list = torch.arange(0, 360, sampling_steps) * torch.pi / 180

    tilt_list = torch.tensor(gaussian_sample(90, tilt_sigma, 7) * torch.pi / 180, dtype=torch.float32)

    # create a (MxN) meshgrid with N rotation around the z axis and M tilt angle

    rot, tilt = torch.meshgrid(rot_list, tilt_list)

    angle_list = torch.stack([rot, tilt], dim=-1)

    sample_shape = angle_list.shape

    # convert the rot tilt psi to angle

    rot_all = angle_list[..., 0].view(-1)
    tilt_all = angle_list[..., 1].view(-1)

    B = len(rot_all)

    all_matrix = []

    for i in range(B):
        ca, sa = torch.cos(rot_all)[i], torch.sin(rot_all)[i]
        cb, sb = torch.cos(tilt_all)[i], torch.sin(tilt_all)[i]

        Ra = torch.tensor([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        Rb = torch.tensor([[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]])
        R = Rb @ Ra
        all_matrix.append(R)

    all_matrix = torch.stack(all_matrix, dim=0)
    all_matrix = all_matrix.reshape((sample_shape[0], sample_shape[1], 3, 3))
    print(all_matrix.shape)

    return all_matrix

def helical_sample_psi(psi_sigma):
    psi_list_1 = torch.tensor(gaussian_sample(0, psi_sigma) * torch.pi / 180, dtype=torch.float32)

    psi_list_2 = torch.tensor(gaussian_sample(180, psi_sigma) * torch.pi / 180, dtype=torch.float32)

    psi_list = torch.cat((psi_list_1, psi_list_2))

    return psi_list
