from astropy.io import fits, ascii
from astropy.wcs import WCS
import astropy.coordinates as coord
import astropy.units as u 
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
from astroquery.vizier import Vizier
import numpy as np 
from tess_stars2px import tess_stars2px_function_entry as tess_point
import pandas as pd 
from astropy.coordinates import Angle
from mk_mass import posterior as mann_19_mass_mk # mann et al. (2019) mass-M_K relation

def query_gaia():
	''' For a given Tierras field, identify stars that are similar types (Teff/logg), on the same chip, and are not variable in TESS'''
	
	PLATE_SCALE = 0.432 

	job = Gaia.launch_job_async('''SELECT  source_id, ra, dec, pmra, pmra_error, pmdec, pmdec_error, parallax, parallax_error, ruwe, radial_velocity, phot_bp_mean_mag, phot_g_mean_mag, phot_rp_mean_mag, bp_rp, phot_variable_flag, non_single_star, teff_gspphot, logg_gspphot, mh_gspphot, r_med_geo, r_lo_geo, r_hi_geo, r_med_photogeo, r_lo_photogeo, r_hi_photogeo,
							 			phot_g_mean_mag - 5 * LOG10(r_med_geo) + 5 AS qg_geo, phot_g_mean_mag - 5 * LOG10(r_med_photogeo) + 5 AS gq_photogeo, 
										original_ext_source_id AS 2MASSID

									FROM ( 
										SELECT * FROM gaiadr3.gaia_source ) 
									AS dr3 
									JOIN external.gaiaedr3_distance using(source_id)
							 		JOIN gaiadr3.tmass_psc_xsc_best_neighbour using(source_id) 

									WHERE ruwe<1.4 
										AND r_med_photogeo <= 50
										AND bp_rp > 1.84
										AND phot_g_mean_mag - 5 * LOG10(r_med_photogeo) + 5 > 6
							 			AND phot_g_mean_mag - 5 * LOG10(r_med_photogeo) + 5 < 16.4
							 			AND dec > -35
								''')
	res = job.get_results()
	return res 

def query_twomass(res, save=True):
	j_mag = np.zeros(len(res))
	h_mag = np.zeros_like(j_mag)
	k_mag = np.zeros_like(j_mag)
	e_j_mag = np.zeros_like(j_mag)
	e_h_mag = np.zeros_like(j_mag)
	e_k_mag = np.zeros_like(j_mag)
	abs_k_mag = np.zeros_like(j_mag)
	e_abs_k_mag = np.zeros_like(j_mag)

	for i in range(len(res)):
		print(f'{i+1} of {len(res)}')
		try:
			v_res = Vizier.query_object('2MASS J'+res['_2MASSID'][i], catalog='II/246', radius=Angle(10,'arcsec'))[0]
		except:
			breakpoint()
		ind = np.where(v_res['_2MASS'] == res['_2MASSID'][i])
		entry = v_res[ind]
		j_mag[i] = entry['Jmag'][0]
		e_j_mag[i] = entry['e_Jmag'][0]
		h_mag[i] = entry['Hmag'][0]
		e_h_mag[i] = entry['e_Hmag'][0]
		k_mag[i] = entry['Kmag'][0]
		e_k_mag[i] = entry['e_Kmag'][0]
	res['j_mag'] = j_mag
	res['j_mag_error'] = e_j_mag
	res['h_mag'] = h_mag
	res['h_mag_error'] = e_h_mag
	res['k_mag'] = k_mag
	res['k_mag_error'] = e_k_mag
	if save:
		res.to_csv('sample.csv',index=0)
	return res

def estimate_m_and_r(res):
	# get masses following M-M_K relation from Benedict et al. (2016)
	abs_k_mags = np.zeros(len(res))
	abs_k_mag_errs = np.zeros_like(abs_k_mags)
	masses = np.zeros_like(abs_k_mags)
	mass_errs = np.zeros_like(abs_k_mags)
	radii = np.zeros_like(abs_k_mags)
	radius_errs = np.zeros_like(abs_k_mags)
	dists = np.zeros_like(abs_k_mags)
	dist_errs = np.zeros_like(abs_k_mags)

	abs_k_mags = res['k_mag'] - 5 * np.log10(res['r_med_photogeo']) + 5
	for i in range(len(res)):
		dist = res['r_med_photogeo'][i]
		# Bailer-Jones errors have some weird values so just use parallaxes for distance errors
		par = res['parallax'][i]
		par_err = res['parallax_error'][i]
		dist_err = par_err/(par**2)
		dists[i] = dist
		dist_errs[i] = dist_err
		k_mag = res['k_mag'][i]
		k_mag_err = res['k_mag_error'][i]
		# abs_k_mags[i] = k_mag - 5*np.log10(dist) + 5
		# abs_k_mag_errs[i] = (k_mag_err**2 + (dist_err*5/(dist*np.log(10)))**2)**0.5
		mass_posterior = mann_19_mass_mk(k_mag, dist, k_mag_err, dist_err)
		masses[i] = np.nanmedian(mass_posterior)	
		mass_errs[i] = np.nanstd(mass_posterior)
	breakpoint()
	return

def exclude_tess(res, save=True):
	# mask out sources that have been/ will be observed in TESS using tess_point
	tess_mask = np.ones(len(res), dtype='bool')
	for i in range(len(res)):
		print(f'{i+1} of {len(res)}')
		outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, outColPix, outRowPix, scinfo = tess_point(1, res['ra'][i], res['dec'][i])
		
		if outSec[0] != -1:
			tess_mask[i] = 0
	res = res[tess_mask]

	if save:
		res_df = res.to_pandas()
		res_df.to_csv('sample.csv',index=0)
	return res

def sample_plots(df):
	mearth_inds = np.where(df['Nexp MEarth'] != 0)[0]

	# plot a cmd 
	plt.scatter(df['bp_rp'], df['gq_photogeo'])
	plt.scatter(df['bp_rp'][mearth_inds], df['gq_photogeo'][mearth_inds])
	plt.gca().invert_yaxis()
	plt.xlabel('Bp-Rp', fontsize=14)
	plt.ylabel('M$_G$', fontsize=14)
	plt.tick_params(labelsize=12)
	plt.tight_layout()
	plt.savefig('plots/cmd.png',dpi=300)

	# do a sky map
	ra = coord.Angle(df['ra'], unit=u.degree)
	dec = coord.Angle(df['dec'], unit=u.degree)
	ra = ra.wrap_at(180*u.degree)

	fig = plt.figure(figsize=(8,6))
	ax = fig.add_subplot(111, projection='mollweide')
	ax.scatter(ra.radian, dec.radian, marker='.')
	ax.scatter(ra.radian[mearth_inds], dec.radian[mearth_inds], marker='.')
	ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
	ax.grid(True)
	plt.tight_layout()
	plt.savefig('plots/sky_plot.png',dpi=300)

	# histogram of apparent magnitudes
	plt.figure()
	rp_bins = np.arange(7, 19, 1)
	plt.hist(df['phot_rp_mean_mag'], bins=rp_bins)
	plt.hist(df['phot_rp_mean_mag'][mearth_inds], bins=rp_bins)
	plt.xlabel('G$_{Rp}$', fontsize=14)
	plt.ylabel('N$_{stars}$', fontsize=14)
	plt.tick_params(labelsize=12)
	plt.tight_layout()
	plt.savefig('plots/rp_hist.png',dpi=300)

	# compare bailer-jones distances to 1/parallax
	plt.figure()
	plt.scatter(1/(df['parallax']/1000),df['r_med_photogeo'])
	plt.scatter(1/(df['parallax'][mearth_inds]/1000),df['r_med_photogeo'][mearth_inds])
	plt.xlabel('1/parallax (pc)', fontsize=14)
	plt.ylabel('d$_{Bailer-Jones} (pc)$', fontsize=14)
	plt.tick_params(labelsize=12)
	plt.tight_layout()

	# construct ra_hms and dec_dms lists
	scs = coord.SkyCoord(df['ra'], df['dec'], unit=(u.degree, u.degree))
	coord_strs = [i.to_string('hmsdms').replace('d',':').replace('h',':').replace('m',':').replace('s','') for i in scs]
	breakpoint()
	return 
    
def exclude_mearth(df):
	# read in the MEarth north/south summary tables 
	# these are available in ascii format at https://lweb.cfa.harvard.edu/MEarth/DR11/north2011-2022/index.html
	# and https://lweb.cfa.harvard.edu/MEarth/DR11/south2014-2022/index.html, respectively

	cols = ['LSPM_Name', 'Gl/GJ', 'LHS', 'rNLTT', 'HIP', '2MASS', 'RA', 'DEC', 'PMRA', 'PMDEC', 'Pi', 'e_Pi', 'Ref', 'SpT', 'Ref', 'B_J', 'R_F', 'I_N', 'J', 'H', 'Ks', 'Mass', 'Rad', 'S', 'Tel', 'Exptime', 'Nmeas', 'Nni', 'Start', 'End']
	north = ascii.read('mearth_north_summary.txt')
	south = ascii.read('mearth_south_summary.txt')
	
	n_mearth_exposures = np.zeros(len(df),dtype='int')
	n_mearth_nights = np.zeros_like(n_mearth_exposures)
	for i in range(len(df)):
		twomass_id = df['_2MASSID'][i]
		ind_n = np.where(north['col6'] == twomass_id)[0]
		ind_s = np.where(south['col6'] == twomass_id)[0]
		if len(ind_n) > 0:
			for j in range(len(ind_n)):
				n_mearth_exposures[i] += int(north['col31'][ind_n[j]])
				n_mearth_nights[i] += int(north['col32'][ind_n[j]])
		if len(ind_s) > 0:
			for j in range(len(ind_s)):
				n_mearth_exposures[i] += int(south['col31'][ind_s[j]])
				n_mearth_nights[i] += int(south['col32'][ind_s[j]])

	df['Nexp MEarth'] = n_mearth_exposures
	df['Nni MEarth'] = n_mearth_nights
	
	return df

if __name__ == '__main__':
	# # step 1: query for nearby M dwarfs in Gaia using colors
	# gaia_res = query_gaia()

	# # step 2: throw out any that have been/will be observed with TESS
	# df = exclude_tess(gaia_res)
	
	# # step 3: figure out which have been observed in MEarth
	# df = pd.read_csv('sample.csv')
	# df = exclude_mearth(df)

	# # step 4: get 2MASS magnitudes 
	# df = query_twomass(df)

	# step 5: estimate masses and radii for the sources
	df = pd.read_csv('sample.csv')
	df = estimate_m_and_r(df)

	breakpoint()

	
	sample_plots(df)
	breakpoint()
	