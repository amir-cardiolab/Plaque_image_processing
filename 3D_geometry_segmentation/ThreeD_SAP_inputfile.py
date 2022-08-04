__author__ = "Mostafa Mahmoudi <mm4238@nau.edu>"
__date__ = "25-06-2021"
__copyright__ = "Copyright (C) 2021 " + __author__
__license__ = "Not Valid"


wdir = '/Users/mostafa/Mostafa_data/Dessertation/Atherosclerosis/Aim2/Quantified_Database2/1A_2012_LAD/'

input_parameters = dict(

	# Directories
	working_dir = wdir,
	output_root = wdir + 'NO/',
	wss_filename = wdir + 'NO/NO_transport_noCAP.vtp',
	CCTA_filename = wdir + '1A_2012_LAD_wall.vtp',
	WSS_LCS_file_name = wdir + '3A_2016_LCx_WSSLCS_cm.vtk',
	SV_pth_filename = wdir + 'LAD.pth',
	transformed_geom_file_name = wdir + 'test.vtp',
	CL_vtk_file_name = wdir + 'Results_old/centerline_split.vtp',



	fieldname = 'wss_average',
	# fieldname = 'f_8_average',

	start_point_ID = 1,
	end_point_ID = 34,
	sector_ave_val_dataset = [],
	bif_point = [175.74, 192.60, -57.55],

	# Flags
	convert2cm_flag = False,
	manual_tangents_flag = True,
	transformed_geom = False,
	bifurcation_set = True,
	SV_centerline = False
	
)