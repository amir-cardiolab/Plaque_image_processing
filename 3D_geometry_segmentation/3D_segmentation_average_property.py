# ***************************************************
# ***** 3D coronary artery model segmentation ******
# ***************************************************
# ***************************************************
# Having the 3D model from CCTA and transformed model
# from CFD simulation, this code transforms the CFD 
# model so that it conforms with the CCTA images.
# The algorithm used is based on Kabsch method.
# Then, having the point IDs of centerline in lesion
# part of artery, this code return the segmented 3D
# model and calculates the average of specified 
# property on each quadrant of each segment.
# ***************************************************
# Developer: Mostafa Mahmoudi
# Start of project: Jan 24, 2021
# Last edit: Mar 1, 2021
# ***************************************************

import numpy
import sys
import vtk
import glob
import re
import math
from vtk.util import numpy_support as VN
from vmtk import pypes
from vmtk import vmtkscripts
import ThreeD_SAP_functions
import ThreeD_SAP_inputfile as inputs

if __name__ == "__main__":

		update_parameters = False
		globals().update(inputs.input_parameters)

		if update_parameters:
			input_parameters.update(

					# Directories
					working_dir = '/Users/mostafa/Mostafa_data/Dessertation/Atherosclerosis/Aim2/Quantified_Database/5A_2010_LCx/Baseline/',
					output_root = working_dir + 'Results/',
					wss_filename = working_dir + 'wss_average_final.vtp',
					CCTA_filename = working_dir + '5A_2010_LCx_wall.vtp',
					WSS_LCS_file_name = wdir + '5A_2010_LCx_WSSLCS_cm.vtk',
					SV_pth_filename = working_dir + 'LCx_new.pth',
					transformed_geom_file_name = working_dir + 'test_cropped.vtp',
					CL_vtk_file_name = working_dir + 'Results_old/centerline_split.vtp',


					fieldname = 'wss_average',
					# fieldname = 'f_8_average'

					start_point_ID = 1,
					end_point_ID = 50,
					sector_ave_val_dataset = [],
					bif_point = [174.9172, 232.9833, -63.3629],

					# Flags
					convert2cm_flag = False,
					manual_tangents_flag = True,
					transformed_geom = True,
					bifurcation_set = True,
					SV_centerline = False

				)

		# output_root = '/Users/mostafa/Mostafa_data/Dessertation/Atherosclerosis/Aim2/Quantified_Database/1A_2012_LCx/Baseline/Results/'
		# wss_filename = '/Users/mostafa/Mostafa_data/Dessertation/Atherosclerosis/Aim2/Quantified_Database/1A_2012_LCx/Baseline/wss_average_final.vtp'
		# CCTA_filename = '/Users/mostafa/Mostafa_data/Dessertation/Atherosclerosis/Aim2/Quantified_Database/1A_2012_LCx/Baseline/1A_2012_LCx_wall.vtp'
		# SV_pth_filename = '/Users/mostafa/Mostafa_data/Dessertation/Atherosclerosis/Aim2/Quantified_Database/1A_2012_LCx/Baseline/LCx_new.pth'

		# fieldname = 'wss_average'
		# # fieldname = 'f_8_average'
		# start_point_ID = 1
		# end_point_ID = 50
		# convert2cm_flag = False
		# manual_tangents_flag = True
		# transformed_geom = True
		# bifurcation_set = True
		# SV_centerline = False
		# sector_ave_val_dataset = []
		# bif_point = [174.9172, 232.9833, -63.3629]


		# extract_centerline(file_name, output_root)
		# read_centerline(output_centerline, output_root)
		# create_segment_locations(file_name, output_root, CL_points)

		if transformed_geom:
			# file_name = '/Users/mostafa/Mostafa_data/Dessertation/Atherosclerosis/Aim2/Quantified_Database/1A_2012_LCx/Baseline/test_cropped.vtp'

			# CL_vtk_file_name = '/Users/mostafa/Mostafa_data/Dessertation/Atherosclerosis/Aim2/Quantified_Database/1A_2012_LCx/Baseline/Results_old/centerline_split.vtp'

			output_CL_coords_path = output_root + 'CL_coords.txt'

			# points_set, CL_vtk = ThreeD_SAP_functions.read_CL_VTK(CL_vtk_file_name)
			output_centerline, output_centerline_vtk, CL_points = ThreeD_SAP_functions.Find_CL_between_points(transformed_geom_file_name, output_root)

			# sys.exit("Exit After Reading VTK Centerline! ")

			print('Loading', transformed_geom_file_name)
			transformed_model_vtkXMLPolyData = vtk.vtkXMLPolyDataReader()
			transformed_model_vtkXMLPolyData.SetFileName(transformed_geom_file_name)
			transformed_model_vtkXMLPolyData.Update()
			# transformed_model_vtkXMLPolyData = reader.GetOutput()
		else:
			wss_centerline, wss_centerline_vtkPolyData = ThreeD_SAP_functions.extract_centerline_vmtk(wss_filename, output_root)
			initial_points = ThreeD_SAP_functions.render_view(wss_centerline_vtkPolyData)

			wss_rotation_points = numpy.transpose(initial_points)

			CCTA_centerline, CCTA_centerline_vtkPolyData = ThreeD_SAP_functions.extract_centerline_vmtk(CCTA_filename, output_root)
			target_points = ThreeD_SAP_functions.render_view(CCTA_centerline_vtkPolyData)

			CCTA_rotation_points = numpy.transpose(target_points)

			R, t = ThreeD_SAP_functions.Kabsch_transformation(wss_rotation_points, CCTA_rotation_points)
			transformation_matrix = ThreeD_SAP_functions.Get_Rotation_translation(R, t)
			print(transformation_matrix)
			transformed_model_vtkXMLPolyData = ThreeD_SAP_functions.transformation_3D_model_matrix(wss_filename, output_root, transformation_matrix)

		if SV_centerline:
			CL_coords, CL_tangents = ThreeD_SAP_functions.read_SV_centerline(SV_pth_filename)
		else:
			CL_coords = CL_points
			# ThreeD_SAP_functions.write_CL(output_CL_coords_path, CL_coords)

		# CL_coords_interp = ThreeD_SAP_functions.point_distribution_along_centerline(CL_coords, 1, True)

		# CL_coords_interp_coarse = ThreeD_SAP_functions.point_distribution_along_centerline(CL_coords_interp, 4, True)

		# print(CL_coords_interp)

		if convert2cm_flag:
			CL_coords_interp = numpy.divide(CL_coords_interp, 10).tolist()

		if bifurcation_set:

			CL_uniform = ThreeD_SAP_functions.split_CL_at_bifurcation(CL_coords, bif_point)

			ThreeD_SAP_functions.write_CL(output_CL_coords_path, CL_uniform)

			# CL_coords_interp = ThreeD_SAP_functions.point_distribution_along_centerline(CL_coords, 1, True)

			CL_coords_interp_coarse = ThreeD_SAP_functions.point_distribution_along_centerline(CL_uniform, 2, True)
			# CL_coords_interp_coarse = ThreeD_SAP_functions.point_distribution_along_centerline(CL_uniform, 2, True)
			print("CL coarse length = ", len(CL_coords_interp_coarse))

			if manual_tangents_flag:
				CL_tangents_interp = ThreeD_SAP_functions.create_tangents(CL_coords_interp_coarse)
				CL_tangents_updated = ThreeD_SAP_functions.update_tangents(CL_tangents_interp, len(CL_uniform))
				print("tan length = ", len(CL_tangents_interp))
				print("tan length updated = ", len(CL_tangents_updated))


		else:

			CL_coords_interp = ThreeD_SAP_functions.point_distribution_along_centerline(CL_coords, 1, True)

			CL_coords_interp_coarse = ThreeD_SAP_functions.point_distribution_along_centerline(CL_coords_interp, 4, True)


			if manual_tangents_flag:
				CL_tangents_interp = ThreeD_SAP_functions.create_tangents(CL_coords_interp_coarse)
				CL_tangents_updated = ThreeD_SAP_functions.update_tangents(CL_tangents_interp)

		print("******************* Updated tangents ******************* ")
		print(CL_tangents_updated)

		end_point_ID = len(CL_uniform) - 3
		for slice_id in range(start_point_ID,end_point_ID):
			# sec1_ave_val, sec2_ave_val, sec3_ave_val, sec4_ave_val = ThreeD_SAP_functions.create_segment_locations(transformed_model_vtkXMLPolyData, CL_coords_interp[slice_id:(slice_id+5)], CL_tangents_updated[int(slice_id/4):(int(slice_id/4)+6)], slice_id, output_root)
			sec1_ave_val, sec2_ave_val, sec3_ave_val, sec4_ave_val = ThreeD_SAP_functions.create_segment_locations(transformed_model_vtkXMLPolyData, CL_uniform[slice_id:(slice_id+2)], CL_tangents_updated[slice_id:(slice_id+2)], slice_id, fieldname, output_root)
			sector_ave_val_dataset.append([sec1_ave_val, sec2_ave_val, sec3_ave_val, sec4_ave_val])
			print("****************** slice %d created! **************" %slice_id)

		ThreeD_SAP_functions.write_output_dataset(output_root+fieldname, sector_ave_val_dataset)






